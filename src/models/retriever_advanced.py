"""
Advanced FAISS retrieval module with IVF and IVF-PQ indices.

Extends the baseline retriever.py with:
- IVF (Inverted File) index for faster search
- IVF-PQ (Product Quantization) for memory-efficient search
- Ground-truth oracle using ROUGE-L with reference summary
- Comprehensive benchmark metrics (Recall@K, MRR, latency, index size)

IMPORTANT: 
- This module is additive - original retriever.py remains unchanged.
- All embeddings MUST be L2 normalized before indexing (for cosine similarity via inner product).
- Chunk order in the list MUST match embedding order when building index.

Reproducibility:
- Set np.random.seed() before training for deterministic results.
- FAISS clustering may have minor variations across runs.

Training Data Requirements:
- IVF: At least 39 * nlist training points recommended
- IVF-PQ: At least max(4096, 16 * nlist) training points for stable codebook
"""

import faiss
import numpy as np
import os
import time
import tempfile
import re
import logging
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer

# Setup logging with basic config if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def _ensure_contiguous_float32(arr: np.ndarray) -> np.ndarray:
    """Ensure array is contiguous float32 for FAISS compatibility."""
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def _normalize_embeddings(embeddings: np.ndarray, verify: bool = True) -> np.ndarray:
    """
    L2 normalize embeddings for cosine similarity via inner product.
    
    This is critical for IndexFlatIP to behave like cosine similarity.
    Normalization is done IN-PLACE for efficiency.
    
    Args:
        embeddings: Float32 embeddings (N, dim)
        verify: If True, assert that normalization succeeded
        
    Returns:
        L2 normalized embeddings (same array, modified in-place)
    """
    embeddings = _ensure_contiguous_float32(embeddings)
    faiss.normalize_L2(embeddings)  # In-place normalization
    
    # Verify normalization succeeded
    if verify:
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), \
            f"Embeddings must be L2-normalized before index. Got norms: [{norms.min():.4f}, {norms.max():.4f}]"
    
    return embeddings


def _get_valid_M_values(dim: int, max_M: Optional[int] = None) -> List[int]:
    """
    Get valid M values (PQ subquantizers) that evenly divide dimension.
    
    Args:
        dim: Embedding dimension
        max_M: Maximum M value to consider (default: dim // 2)
        
    Returns:
        List of valid M values, sorted ascending
    """
    if max_M is None:
        max_M = dim // 2
    
    # Find all divisors of dim that are reasonable for PQ
    valid_Ms = [m for m in range(1, max_M + 1) if dim % m == 0 and m <= dim // 2]
    
    # Filter to reasonable values (at least 8 for meaningful quantization)
    valid_Ms = [m for m in valid_Ms if m >= 8]
    
    if not valid_Ms:
        # Fallback: using M=dim will disable PQ (store full vectors)
        logger.warning(f"No small M divides dim={dim}; using M=dim which effectively disables PQ.")
        return [dim]
    
    return sorted(valid_Ms)


def _select_best_M(dim: int, requested_M: int) -> int:
    """
    Select best valid M value closest to requested.
    
    Args:
        dim: Embedding dimension
        requested_M: Desired M value
        
    Returns:
        Valid M value that evenly divides dim
    """
    valid_Ms = _get_valid_M_values(dim)
    
    if not valid_Ms or len(valid_Ms) == 0:
        logger.warning(f"No suitable M dividing dim={dim}; using M={dim} (PQ effectively disabled). Consider changing requested_M.")
        print(f"  ⚠️ No suitable M dividing dim={dim}; using M={dim} (PQ effectively disabled). Consider changing requested_M.")
        return dim
    
    best_M = min(valid_Ms, key=lambda x: abs(x - requested_M))
    
    if best_M != requested_M:
        print(f"  ⚠️ Adjusted M from {requested_M} to {best_M} (dim={dim} must be divisible by M)")
    
    return best_M


def set_reproducibility_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Note: FAISS internal clustering may have minor variations across runs.
    For full determinism, consider FAISS's faiss.ParameterSpace() if available.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    logger.debug(f"Set reproducibility seed: {seed}")


# =============================================================================
# FAISS Index Builders
# =============================================================================

def build_flat_index(embeddings: np.ndarray, normalize: bool = True) -> faiss.Index:
    """
    Build baseline FAISS Flat index (exact search).
    
    Args:
        embeddings: Float32 embeddings (N, dim)
        normalize: If True, L2 normalize embeddings (required for cosine similarity)
        
    Returns:
        FAISS IndexFlatIP
        
    Note:
        - Embeddings are normalized in-place if normalize=True
        - For cosine similarity, MUST use normalized embeddings with IndexFlatIP
        - Embedding order MUST match chunk order in your list
    """
    embeddings = _ensure_contiguous_float32(embeddings)
    
    if normalize:
        embeddings = _normalize_embeddings(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    return index


def build_ivf_index(
    embeddings: np.ndarray,
    nlist: int = 1024,
    nprobe: int = 8,
    normalize: bool = True,
    seed: int = 42
) -> faiss.Index:
    """
    Build FAISS IVF (Inverted File) index.
    
    IVF partitions the embedding space into nlist clusters.
    At search time, only nprobe clusters are searched.
    
    Args:
        embeddings: Float32 embeddings (N, dim)
        nlist: Number of clusters (voronoi cells)
        nprobe: Number of clusters to search at query time
        normalize: If True, L2 normalize embeddings
        seed: Random seed for reproducibility
        
    Returns:
        Trained FAISS IndexIVFFlat
        
    Note:
        - nlist should be roughly sqrt(N) to 4*sqrt(N)
        - Higher nprobe = higher recall but slower search
        - FAISS recommends at least 39 * nlist training points
        - Embedding order MUST match chunk order
    """
    set_reproducibility_seed(seed)
    
    embeddings = _ensure_contiguous_float32(embeddings)
    if normalize:
        embeddings = _normalize_embeddings(embeddings)
    
    n_vectors, dim = embeddings.shape
    
    # FAISS requires sufficient training points per cluster
    min_points_per_cluster = 39
    max_nlist = max(1, n_vectors // min_points_per_cluster)
    original_nlist = nlist
    nlist = min(nlist, max_nlist, n_vectors)
    nlist = max(1, nlist)
    
    if nlist != original_nlist:
        print(f"  ⚠️ Adjusted nlist from {original_nlist} to {nlist} (need {min_points_per_cluster}*nlist training points)")
    
    # If dataset is too small for IVF, raise error instead of fallback
    min_vectors = 64  # Lowered from 100
    if n_vectors < min_vectors:
        raise ValueError(
            f"Too few vectors ({n_vectors}) for IVF training. Need at least {min_vectors}. "
            f"Use more samples or smaller chunk_size."
        )
    
    # Create quantizer (flat index for cluster centroids)
    quantizer = faiss.IndexFlatIP(dim)
    
    # Create IVF index
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train on embeddings (learns cluster centroids)
    try:
        index.train(embeddings)
    except Exception as e:
        logger.error(f"IVF training failed: {e}")
        raise RuntimeError(f"IVF training failed: {e}") from e
    
    # Add embeddings
    index.add(embeddings)
    
    # Set nprobe for search (cannot exceed nlist)
    index.nprobe = min(nprobe, nlist)
    
    return index


def build_ivf_pq_index(
    embeddings: np.ndarray,
    nlist: int = 1024,
    M: int = 96,
    nbits: int = 8,
    nprobe: int = 8,
    normalize: bool = True,
    seed: int = 42
) -> faiss.Index:
    """
    Build FAISS IVF-PQ (Inverted File with Product Quantization) index.
    
    Combines IVF clustering with PQ compression for memory efficiency.
    
    Args:
        embeddings: Float32 embeddings (N, dim)
        nlist: Number of IVF clusters
        M: Number of PQ subquantizers (dim must be divisible by M)
        nbits: Bits per PQ code (4 or 8)
        nprobe: Number of clusters to search
        normalize: If True, L2 normalize embeddings
        seed: Random seed for reproducibility
        
    Returns:
        Trained FAISS IndexIVFPQ
        
    Note:
        Memory per vector = M * nbits / 8 bytes
        - M=96, nbits=8 → 96 bytes/vector
        - M=96, nbits=4 → 48 bytes/vector
        - M=48, nbits=8 → 48 bytes/vector
        - M=48, nbits=4 → 24 bytes/vector
        
        Training requirements:
        - PQ codebook needs sufficient training data (ideally ≥ 256 * 2^nbits)
        - FAISS requires n_vectors >= nlist for IVF training
        - Embedding order MUST match chunk order
    """
    set_reproducibility_seed(seed)
    
    embeddings = _ensure_contiguous_float32(embeddings)
    if normalize:
        embeddings = _normalize_embeddings(embeddings)
    
    n_vectors, dim = embeddings.shape
    
    # Select valid M that divides dim evenly
    M = _select_best_M(dim, M)
    
    # PQ training requirements - conservative but practical formula
    # Rule: at least max(4096, 16 * nlist) for stable PQ codebook training
    min_pq_training = max(4096, 16 * nlist)
    min_points_per_cluster = 39
    
    # Log training data sufficiency
    if n_vectors < min_pq_training:
        print(f"  ⚠️ PQ training: {n_vectors} vectors, recommended ≥{min_pq_training} for stable PQ training")
        logger.warning(f"PQ training: {n_vectors} vectors < recommended {min_pq_training}")
    
    # Adjust nlist to ensure sufficient training points
    max_nlist = max(1, n_vectors // min_points_per_cluster)
    original_nlist = nlist
    nlist = min(nlist, max_nlist, n_vectors)
    nlist = max(1, nlist)
    
    if nlist != original_nlist:
        print(f"  ⚠️ Adjusted nlist from {original_nlist} to {nlist} (need {min_points_per_cluster}*nlist training points)")
    
    # If dataset is too small for IVF-PQ, raise error instead of fallback
    min_vectors = 64  # Lowered from 256
    if n_vectors < min_vectors:
        raise ValueError(
            f"Too few vectors ({n_vectors}) for IVF-PQ training. Need at least {min_vectors}. "
            f"Use more samples or smaller chunk_size."
        )
    
    # Create quantizer
    quantizer = faiss.IndexFlatIP(dim)
    
    # Create IVF-PQ index
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, M, nbits)
    
    # Train on embeddings with error handling
    try:
        index.train(embeddings)
    except Exception as e:
        logger.error(f"IVF-PQ training failed: {e}")
        raise RuntimeError(f"IVF-PQ training failed: {e}") from e
    
    # Add embeddings
    index.add(embeddings)
    
    # Set nprobe (cannot exceed nlist)
    index.nprobe = min(nprobe, nlist)
    
    return index


def set_nprobe(index: faiss.Index, nprobe: int) -> None:
    """
    Set nprobe parameter for IVF-based indices.
    
    Args:
        index: FAISS index (IVF or IVF-PQ)
        nprobe: Number of clusters to search
    """
    if hasattr(index, 'nprobe'):
        index.nprobe = nprobe


# =============================================================================
# Ground-Truth Oracle
# =============================================================================

def compute_ground_truth_chunks(
    chunks: List[Dict],
    reference_summary: str,
    top_n: int = 5
) -> Tuple[List[int], List[float]]:
    """
    Compute ground-truth chunk indices using ROUGE-L with reference summary.
    
    Strategy:
    - For each chunk, compute ROUGE-L F1 with gold summary
    - Rank chunks by ROUGE-L score
    - Top-N chunks become ground-truth
    
    This is methodologically sound because:
    - Reference summary is model-independent
    - No leakage from retrieval model
    - Academically acceptable oracle
    
    Args:
        chunks: List of chunk dicts with 'text' key
                NOTE: Chunk order MUST match embedding/index order
        reference_summary: Gold standard summary from dataset
        top_n: Number of top chunks to consider ground-truth
        
    Returns:
        Tuple of (ground_truth_indices, rouge_scores)
        Indices correspond to positions in chunks list (and thus index)
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # Compute ROUGE-L for each chunk
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get("text", "")
        if not chunk_text.strip():
            chunk_scores.append((i, 0.0))
            continue
            
        scores = scorer.score(reference_summary, chunk_text)
        rouge_l_f1 = scores['rougeL'].fmeasure
        chunk_scores.append((i, rouge_l_f1))
    
    # Sort by ROUGE-L score (descending)
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top-N as ground truth
    ground_truth_indices = [idx for idx, _ in chunk_scores[:top_n]]
    ground_truth_scores = [score for _, score in chunk_scores[:top_n]]
    
    return ground_truth_indices, ground_truth_scores


# =============================================================================
# Benchmark Metrics
# =============================================================================

def compute_recall_at_k(
    retrieved_indices: List[int],
    ground_truth_indices: List[int],
    k: int
) -> Optional[float]:
    """
    Compute Recall@K.
    
    Recall@K = |retrieved[:k] ∩ ground_truth| / |ground_truth|
    
    Args:
        retrieved_indices: Indices returned by retrieval model (ordered by score)
        ground_truth_indices: Oracle ground-truth indices
        k: Number of top retrieved to consider
        
    Returns:
        Recall@K value in [0, 1], or None if ground_truth is empty
        (caller should handle None as excluded from aggregation)
    """
    if not ground_truth_indices:
        logger.debug("Empty ground_truth_indices, returning None for Recall@K")
        return None
    
    retrieved_top_k = set(retrieved_indices[:k])
    ground_truth_set = set(ground_truth_indices)
    
    intersection = retrieved_top_k & ground_truth_set
    recall = len(intersection) / len(ground_truth_set)
    
    return recall


def compute_mrr(
    retrieved_indices: List[int],
    ground_truth_indices: List[int]
) -> Optional[float]:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    MRR = 1 / rank of first relevant document
    
    Args:
        retrieved_indices: Indices returned by retrieval model (ordered by score)
        ground_truth_indices: Oracle ground-truth indices
        
    Returns:
        MRR value in [0, 1], or None if ground_truth is empty
        (caller should handle None as excluded from aggregation)
    """
    if not ground_truth_indices:
        logger.debug("Empty ground_truth_indices, returning None for MRR")
        return None
    
    ground_truth_set = set(ground_truth_indices)
    
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in ground_truth_set:
            return 1.0 / rank
    
    return 0.0  # No relevant document found


def measure_query_latency(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    k: int,
    n_queries: int = 100
) -> Dict[str, float]:
    """
    Measure query latency statistics.
    
    Args:
        index: FAISS index
        query_embeddings: Query embeddings to use (will sample if > n_queries)
        k: Number of results to retrieve
        n_queries: Number of queries to run for statistics
        
    Returns:
        Dict with mean, p50, p90, p95 latency in milliseconds
    """
    query_embeddings = _ensure_contiguous_float32(query_embeddings)
    
    # Sample queries if needed
    if len(query_embeddings) > n_queries:
        indices = np.random.choice(len(query_embeddings), n_queries, replace=False)
        query_embeddings = query_embeddings[indices]
    
    latencies = []
    for q_emb in query_embeddings:
        q_emb = q_emb.reshape(1, -1)
        
        start = time.perf_counter()
        _, _ = index.search(q_emb, k)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p95_ms": float(np.percentile(latencies, 95))
    }


def get_index_size_mb(index: faiss.Index) -> float:
    """
    Get FAISS index size in megabytes.
    
    Uses mkstemp for cross-platform temp file handling.
    
    Args:
        index: FAISS index
        
    Returns:
        Size in MB
    """
    # Use mkstemp for cross-platform compatibility (avoids Windows file locking)
    fd, temp_path = tempfile.mkstemp(suffix='.faiss')
    try:
        os.close(fd)  # Close file descriptor, FAISS will open it
        faiss.write_index(index, temp_path)
        size_bytes = os.path.getsize(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    return size_bytes / (1024 * 1024)


# =============================================================================
# Retrieval with Benchmarking
# =============================================================================

def retrieve_and_benchmark(
    query_text: str,
    chunks: List[Dict],
    index: faiss.Index,
    embedding_model: SentenceTransformer,
    k: int,
    ground_truth_indices: List[int]
) -> Dict:
    """
    Retrieve top-K chunks and compute benchmark metrics.
    
    Args:
        query_text: Query text
        chunks: List of chunk dicts (order MUST match index order)
        index: FAISS index
        embedding_model: Sentence transformer model
        k: Number of chunks to retrieve
        ground_truth_indices: Oracle ground-truth chunk indices
        
    Returns:
        Dict with retrieved indices, scores, and metrics
        
    Note:
        Query embedding is L2 normalized to match index embeddings.
    """
    # Embed query with version compatibility fallback
    try:
        query_emb = embedding_model.encode(
            [query_text],
            convert_to_numpy=True,
            show_progress_bar=False
        )
    except TypeError:
        # Fallback for older sentence-transformers versions
        query_emb = np.array(embedding_model.encode([query_text], show_progress_bar=False))
    
    query_emb = _ensure_contiguous_float32(query_emb)
    
    # Manually normalize (don't rely on normalize_embeddings parameter)
    faiss.normalize_L2(query_emb)
    
    # Dimension check: ensure query embedding matches index dimension
    if query_emb.shape[1] != index.d:
        raise ValueError(
            f"Query embedding dim {query_emb.shape[1]} != index dim {index.d}. "
            f"Check embeddings/index consistency (same embedding model?)."
        )
    
    # Search
    num_chunks = len(chunks)
    k_clamped = min(k, num_chunks)
    
    start = time.perf_counter()
    scores, indices = index.search(query_emb, k_clamped)
    latency_ms = (time.perf_counter() - start) * 1000
    
    # Filter valid indices
    retrieved_indices = []
    retrieved_scores = []
    for idx, score in zip(indices[0], scores[0]):
        if 0 <= idx < num_chunks:
            retrieved_indices.append(int(idx))
            retrieved_scores.append(float(score))
    
    # Compute metrics (may return None if ground_truth is empty)
    recall_1 = compute_recall_at_k(retrieved_indices, ground_truth_indices, 1)
    recall_5 = compute_recall_at_k(retrieved_indices, ground_truth_indices, 5)
    recall_10 = compute_recall_at_k(retrieved_indices, ground_truth_indices, 10)
    mrr = compute_mrr(retrieved_indices, ground_truth_indices)
    
    # Flag indicating if ground-truth was valid (for aggregation filtering)
    gt_valid = ground_truth_indices is not None and len(ground_truth_indices) > 0
    
    return {
        "retrieved_indices": retrieved_indices,
        "retrieved_scores": retrieved_scores,
        "recall_at_1": recall_1 if recall_1 is not None else 0.0,
        "recall_at_5": recall_5 if recall_5 is not None else 0.0,
        "recall_at_10": recall_10 if recall_10 is not None else 0.0,
        "mrr": mrr if mrr is not None else 0.0,
        "latency_ms": latency_ms,
        "gt_valid": gt_valid
    }


# =============================================================================
# Index Caching
# =============================================================================

def _sanitize_filename(s: str) -> str:
    """Sanitize string for use in filename (remove/replace invalid chars)."""
    # Replace invalid filename chars with underscore
    return re.sub(r'[<>:"/\\|?*]', '_', str(s))


def save_index_with_metadata(
    index: faiss.Index,
    cache_dir: str,
    cache_key: str,
    index_type: str,
    params: Dict
) -> str:
    """
    Save FAISS index with metadata for reproducibility.
    
    Args:
        index: FAISS index
        cache_dir: Cache directory
        cache_key: Base cache key (from embeddings)
        index_type: 'flat', 'ivf', or 'ivf_pq'
        params: Index parameters (nlist, M, nbits, nprobe)
        
    Returns:
        Path to saved index
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create unique filename with parameters (sanitized)
    param_str = "_".join(f"{_sanitize_filename(k)}{_sanitize_filename(v)}" for k, v in sorted(params.items()))
    filename = f"index_{_sanitize_filename(index_type)}_{param_str}_{_sanitize_filename(cache_key)}.faiss"
    cache_path = os.path.join(cache_dir, filename)
    
    faiss.write_index(index, cache_path)
    
    return cache_path


def load_index_with_metadata(
    cache_dir: str,
    cache_key: str,
    index_type: str,
    params: Dict
) -> Optional[faiss.Index]:
    """
    Load FAISS index from cache.
    
    Args:
        cache_dir: Cache directory
        cache_key: Base cache key
        index_type: 'flat', 'ivf', or 'ivf_pq'
        params: Index parameters
        
    Returns:
        FAISS index or None if not found
    """
    # Use same sanitization as save_index_with_metadata for consistency
    param_str = "_".join(f"{_sanitize_filename(k)}{_sanitize_filename(v)}" for k, v in sorted(params.items()))
    filename = f"index_{_sanitize_filename(index_type)}_{param_str}_{_sanitize_filename(cache_key)}.faiss"
    cache_path = os.path.join(cache_dir, filename)
    
    if os.path.exists(cache_path):
        return faiss.read_index(cache_path)
    
    return None


# =============================================================================
# Index Info
# =============================================================================

def get_index_info(index: faiss.Index) -> Dict:
    """
    Get information about FAISS index.
    
    Uses class name checking for robustness across FAISS versions.
    
    Args:
        index: FAISS index
        
    Returns:
        Dict with index type, parameters, size
    """
    info = {
        "ntotal": index.ntotal,
        "d": index.d,
        "is_trained": index.is_trained
    }
    
    # Use class name for more robust type detection
    class_name = index.__class__.__name__
    
    if 'IndexFlatIP' in class_name or 'IndexFlat' in class_name:
        info["type"] = "flat"
    elif 'IndexIVFPQ' in class_name or hasattr(index, 'pq'):
        info["type"] = "ivf_pq"
        if hasattr(index, 'nlist'):
            info["nlist"] = index.nlist
        if hasattr(index, 'nprobe'):
            info["nprobe"] = index.nprobe
        if hasattr(index, 'pq'):
            info["M"] = index.pq.M
            info["nbits"] = index.pq.nbits
    elif 'IndexIVF' in class_name:
        info["type"] = "ivf"
        if hasattr(index, 'nlist'):
            info["nlist"] = index.nlist
        if hasattr(index, 'nprobe'):
            info["nprobe"] = index.nprobe
    else:
        info["type"] = "unknown"
    
    info["size_mb"] = get_index_size_mb(index)
    
    return info
