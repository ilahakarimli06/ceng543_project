"""
PLAID-style Token Pruning and Compression Retrieval.

Implements PLAID optimizations for ColBERT:
- Token importance scoring (TF-IDF, attention-based)
- Token pruning (keep top-T tokens per chunk)
- Product Quantization compression for token embeddings

Key insight:
- ColBERT stores all tokens → large index
- PLAID prunes unimportant tokens + compresses → smaller index
- Trade-off: slight recall loss for significant memory savings

Usage:
    colbert_model, colbert_tok = load_colbert_model()
    colbert_index = build_token_index(chunks, colbert_model, colbert_tok)
    plaid_index = build_plaid_index(colbert_index, chunks, top_t=32, M=48, nbits=8)
    results = plaid_retrieve(query, plaid_index, colbert_model, colbert_tok, top_k=10)

Reference:
    Santhanam et al. "PLAID: An Efficient Engine for Late Interaction Retrieval"
    (CIKM 2022)
"""

import numpy as np
import faiss
import time
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer

# Import from colbert_retriever
from src.models.colbert_retriever import (
    ColBERTIndex, 
    load_colbert_model, 
    embed_tokens,
    get_device
)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PLAIDIndex:
    """PLAID index with pruned and compressed token embeddings."""
    # Compressed embeddings (after PQ)
    token_embeddings: np.ndarray     # (N_pruned_tokens, dim) or PQ codes
    token_to_chunk: np.ndarray       # (N_pruned_tokens,) - token_idx → chunk_idx
    chunk_token_ranges: List[Tuple[int, int]]  # chunk_idx → (start, end)
    
    # PQ index for compressed search
    pq_index: Optional[faiss.Index]  # FAISS IndexPQ or IndexIVFPQ
    
    # Metadata
    dim: int
    num_chunks: int
    num_tokens: int                  # Pruned token count
    tokens_per_chunk: int            # T value used
    M: int                           # PQ subquantizers
    nbits: int                       # Bits per subquantizer
    compression_ratio: float         # Original / compressed size


@dataclass
class TokenImportance:
    """Token importance scores for a chunk."""
    chunk_idx: int
    token_indices: np.ndarray   # Original indices in chunk
    importance_scores: np.ndarray
    token_embeddings: np.ndarray


# =============================================================================
# Token Importance Scoring
# =============================================================================

def compute_tfidf_importance(
    chunks: List[Dict],
    tokenizer
) -> Dict[int, np.ndarray]:
    """
    Compute TF-IDF based token importance.
    
    Strategy:
    - Build TF-IDF vectorizer over all chunks
    - For each token in each chunk, score = TF-IDF weight
    
    Args:
        chunks: List of chunk dicts with 'text' key
        tokenizer: HuggingFace tokenizer for consistent tokenization
        
    Returns:
        Dict mapping chunk_idx → (token_count,) importance array
    """
    # Build corpus for TF-IDF
    texts = [chunk["text"] for chunk in chunks]
    
    # Use simple whitespace tokenization for TF-IDF
    # (matches better with BERT subword tokens in spirit)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=10000
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    word_to_tfidf = {}
    
    # Map words to their max TF-IDF across documents
    for doc_idx, text in enumerate(texts):
        row = tfidf_matrix[doc_idx].toarray().flatten()
        for word_idx, score in enumerate(row):
            if score > 0:
                word = feature_names[word_idx]
                word_to_tfidf[word] = max(word_to_tfidf.get(word, 0), score)
    
    # Map to token-level importance
    chunk_importance = {}
    
    for chunk_idx, chunk in enumerate(chunks):
        text = chunk["text"].lower()
        # Tokenize using provided tokenizer
        tokens = tokenizer.tokenize(text)
        
        # Score each token
        scores = []
        for token in tokens:
            # Handle subword tokens (remove ## prefix for BERT)
            clean_token = token.replace("##", "").lower()
            # Find best matching word
            best_score = 0
            for word, score in word_to_tfidf.items():
                if clean_token in word or word in clean_token:
                    best_score = max(best_score, score)
            scores.append(best_score)
        
        chunk_importance[chunk_idx] = np.array(scores, dtype=np.float32)
    
    return chunk_importance


def compute_positional_importance(
    num_tokens: int,
    decay: float = 0.95
) -> np.ndarray:
    """
    Compute position-based importance (earlier tokens more important).
    
    Args:
        num_tokens: Number of tokens
        decay: Decay factor per position
        
    Returns:
        (num_tokens,) importance array
    """
    positions = np.arange(num_tokens)
    return decay ** positions


def compute_hybrid_importance(
    chunks: List[Dict],
    colbert_index: ColBERTIndex,
    tokenizer,
    tfidf_weight: float = 0.7,
    position_weight: float = 0.3
) -> Dict[int, np.ndarray]:
    """
    Compute hybrid importance combining TF-IDF and positional scores.
    
    Args:
        chunks: List of chunk dicts
        colbert_index: ColBERT index with token embeddings
        tokenizer: Tokenizer for TF-IDF tokenization
        tfidf_weight: Weight for TF-IDF component
        position_weight: Weight for positional component
        
    Returns:
        Dict mapping chunk_idx → importance array
    """
    # Get TF-IDF importance
    tfidf_scores = compute_tfidf_importance(chunks, tokenizer)
    
    # Combine with positional importance
    chunk_importance = {}
    
    for chunk_idx, (start, end) in enumerate(colbert_index.chunk_token_ranges):
        chunk_len = end - start
        if chunk_len == 0:
            chunk_importance[chunk_idx] = np.array([], dtype=np.float32)
            continue
        
        # Get TF-IDF scores (may not match exactly due to special tokens)
        if chunk_idx in tfidf_scores:
            tfidf = tfidf_scores[chunk_idx]
            # Pad or truncate to match chunk length
            if len(tfidf) < chunk_len:
                tfidf = np.pad(tfidf, (0, chunk_len - len(tfidf)), constant_values=0)
            else:
                tfidf = tfidf[:chunk_len]
        else:
            tfidf = np.zeros(chunk_len, dtype=np.float32)
        
        # Get positional importance
        pos_importance = compute_positional_importance(chunk_len)
        
        # Normalize and combine
        if tfidf.max() > 0:
            tfidf = tfidf / tfidf.max()
        
        combined = tfidf_weight * tfidf + position_weight * pos_importance
        chunk_importance[chunk_idx] = combined
    
    return chunk_importance


# =============================================================================
# Token Pruning
# =============================================================================

def prune_tokens(
    colbert_index: ColBERTIndex,
    importance_scores: Dict[int, np.ndarray],
    tokens_per_chunk: int = 32
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Prune tokens to keep only top-T per chunk.
    
    Args:
        colbert_index: Full ColBERT index
        importance_scores: Dict mapping chunk_idx → importance array
        tokens_per_chunk: Number of tokens to keep per chunk (T)
        
    Returns:
        Tuple of (pruned_embeddings, pruned_token_to_chunk, pruned_ranges)
    """
    pruned_embeddings = []
    pruned_token_to_chunk = []
    pruned_ranges = []
    
    current_idx = 0
    
    for chunk_idx, (start, end) in enumerate(colbert_index.chunk_token_ranges):
        chunk_len = end - start
        
        if chunk_len == 0:
            pruned_ranges.append((current_idx, current_idx))
            continue
        
        # Get importance scores for this chunk
        if chunk_idx in importance_scores:
            scores = importance_scores[chunk_idx]
            if len(scores) != chunk_len:
                # Fallback: uniform importance
                scores = np.ones(chunk_len, dtype=np.float32)
        else:
            scores = np.ones(chunk_len, dtype=np.float32)
        
        # Select top-T tokens
        t = min(tokens_per_chunk, chunk_len)
        top_indices = np.argsort(scores)[::-1][:t]
        top_indices = np.sort(top_indices)  # Preserve order
        
        # Extract embeddings
        chunk_start = start
        for rel_idx in top_indices:
            abs_idx = chunk_start + rel_idx
            pruned_embeddings.append(colbert_index.token_embeddings[abs_idx])
            pruned_token_to_chunk.append(chunk_idx)
        
        pruned_ranges.append((current_idx, current_idx + len(top_indices)))
        current_idx += len(top_indices)
    
    if pruned_embeddings:
        pruned_emb = np.vstack(pruned_embeddings).astype(np.float32)
    else:
        pruned_emb = np.zeros((0, colbert_index.dim), dtype=np.float32)
    
    return (
        pruned_emb,
        np.array(pruned_token_to_chunk, dtype=np.int32),
        pruned_ranges
    )


# =============================================================================
# PQ Compression
# =============================================================================

def build_pq_index(
    embeddings: np.ndarray,
    M: int = 48,
    nbits: int = 8,
    seed: int = 42
) -> faiss.Index:
    """
    Build Product Quantization index for compressed search.
    
    Args:
        embeddings: (N, dim) float32 embeddings
        M: Number of subquantizers (dim must be divisible by M)
        nbits: Bits per subquantizer code
        seed: Random seed for reproducibility
        
    Returns:
        Trained FAISS IndexPQ
        
    Raises:
        ValueError: If too few vectors for PQ training
    """
    np.random.seed(seed)
    
    n_vectors, dim = embeddings.shape
    
    # Adjust M to divide dim evenly
    original_M = M
    while dim % M != 0 and M > 1:
        M -= 1
    if M != original_M:
        print(f"    ⚠️ Adjusted M from {original_M} to {M} (dim={dim} must be divisible by M)")
    
    # Minimum 64 vectors for PQ (lowered from 256)
    min_vectors = 64
    if n_vectors < min_vectors:
        raise ValueError(
            f"Too few vectors ({n_vectors}) for PQ training. Need at least {min_vectors}. "
            f"Increase tokens_per_chunk or use more samples."
        )
    
    # Create PQ index
    index = faiss.IndexPQ(dim, M, nbits)
    
    # Train
    index.train(embeddings)
    
    # Add
    index.add(embeddings)
    
    return index


def build_ivf_pq_token_index(
    embeddings: np.ndarray,
    nlist: int = 256,
    M: int = 48,
    nbits: int = 8,
    nprobe: int = 8,
    seed: int = 42
) -> faiss.Index:
    """
    Build IVF-PQ index for faster compressed search.
    
    Args:
        embeddings: (N, dim) float32 embeddings
        nlist: Number of IVF clusters
        M: Number of PQ subquantizers
        nbits: Bits per code
        nprobe: Clusters to search at query time
        seed: Random seed
        
    Returns:
        Trained FAISS IndexIVFPQ
        
    Raises:
        ValueError: If too few vectors for IVF-PQ training
    """
    np.random.seed(seed)
    
    n_vectors, dim = embeddings.shape
    
    # Adjust M to divide dim evenly
    original_M = M
    while dim % M != 0 and M > 1:
        M -= 1
    if M != original_M:
        print(f"    ⚠️ Adjusted M from {original_M} to {M} (dim={dim})")
    
    # Adjust nlist based on dataset size
    min_per_cluster = 39
    max_nlist = max(1, n_vectors // min_per_cluster)
    original_nlist = nlist
    nlist = min(nlist, max_nlist, n_vectors)
    nlist = max(1, nlist)
    
    if nlist != original_nlist:
        print(f"    ⚠️ Adjusted nlist from {original_nlist} to {nlist}")
    
    # Minimum 64 vectors for IVF-PQ
    min_vectors = 64
    if n_vectors < min_vectors:
        raise ValueError(
            f"Too few vectors ({n_vectors}) for IVF-PQ training. Need at least {min_vectors}. "
            f"Increase tokens_per_chunk or use more samples."
        )
    
    # Create quantizer
    quantizer = faiss.IndexFlatIP(dim)
    
    # Create IVF-PQ
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, M, nbits)
    
    # Train
    index.train(embeddings)
    
    # Add
    index.add(embeddings)
    
    # Set nprobe
    index.nprobe = min(nprobe, nlist)
    
    return index


# =============================================================================
# PLAID Index Building
# =============================================================================

def build_plaid_index(
    colbert_index: ColBERTIndex,
    chunks: List[Dict],
    tokenizer,
    tokens_per_chunk: int = 32,
    M: int = 48,
    nbits: int = 8,
    use_ivf: bool = True,
    nlist: int = 256,
    nprobe: int = 8,
    seed: int = 42
) -> PLAIDIndex:
    """
    Build PLAID index with token pruning and PQ compression.
    
    Args:
        colbert_index: Full ColBERT index
        chunks: Original chunks for importance computation
        tokenizer: Tokenizer for TF-IDF
        tokens_per_chunk: T - tokens to keep per chunk
        M: PQ subquantizers
        nbits: Bits per subquantizer
        use_ivf: If True, use IVF-PQ instead of plain PQ
        nlist: IVF clusters
        nprobe: IVF search clusters
        seed: Random seed
        
    Returns:
        PLAIDIndex with pruned and compressed tokens
    """
    print(f"  🔧 Building PLAID index: T={tokens_per_chunk}, M={M}, nbits={nbits}")
    
    # Step 1: Compute token importance
    print(f"    Computing token importance...")
    importance_scores = compute_hybrid_importance(
        chunks, colbert_index, tokenizer
    )
    
    # Step 2: Prune tokens
    print(f"    Pruning to {tokens_per_chunk} tokens per chunk...")
    pruned_emb, pruned_map, pruned_ranges = prune_tokens(
        colbert_index, importance_scores, tokens_per_chunk
    )
    
    original_tokens = colbert_index.num_tokens
    pruned_tokens = len(pruned_emb)
    prune_ratio = pruned_tokens / original_tokens if original_tokens > 0 else 0
    print(f"    Pruned: {original_tokens} → {pruned_tokens} tokens ({prune_ratio:.1%})")
    
    # Step 3: Build PQ index
    print(f"    Building PQ index (M={M}, nbits={nbits})...")
    
    if pruned_tokens > 0:
        # Normalize embeddings for inner product search
        faiss.normalize_L2(pruned_emb)
        
        if use_ivf and pruned_tokens >= 256:
            pq_index = build_ivf_pq_token_index(
                pruned_emb, nlist, M, nbits, nprobe, seed
            )
        else:
            pq_index = build_pq_index(pruned_emb, M, nbits, seed)
    else:
        pq_index = None
    
    # Calculate compression ratio
    original_size = colbert_index.num_tokens * colbert_index.dim * 4  # float32
    
    if pq_index is not None:
        # Compressed size: tokens × M × nbits / 8 bytes
        compressed_size = pruned_tokens * M * nbits // 8
        # Add IVF overhead if applicable
        if hasattr(pq_index, 'nlist'):
            compressed_size += pq_index.nlist * colbert_index.dim * 4
    else:
        compressed_size = original_size
    
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    print(f"    Compression ratio: {compression_ratio:.1f}x")
    
    return PLAIDIndex(
        token_embeddings=pruned_emb,
        token_to_chunk=pruned_map,
        chunk_token_ranges=pruned_ranges,
        pq_index=pq_index,
        dim=colbert_index.dim,
        num_chunks=colbert_index.num_chunks,
        num_tokens=pruned_tokens,
        tokens_per_chunk=tokens_per_chunk,
        M=M,
        nbits=nbits,
        compression_ratio=compression_ratio
    )


# =============================================================================
# PLAID Retrieval
# =============================================================================

def plaid_retrieve(
    query_text: str,
    plaid_index: PLAIDIndex,
    model,
    tokenizer,
    top_k: int = 10,
    max_query_length: int = 64
) -> Tuple[List[int], List[float], float]:
    """
    Retrieve using PLAID index.
    
    Strategy:
    1. Embed query tokens
    2. For each query token, search PQ index for similar document tokens
    3. Aggregate scores by chunk (approximate MaxSim)
    
    Args:
        query_text: Query text
        plaid_index: PLAID index
        model: BERT model
        tokenizer: BERT tokenizer
        top_k: Chunks to retrieve
        max_query_length: Maximum query tokens
        
    Returns:
        Tuple of (indices, scores, latency_ms)
    """
    start_time = time.perf_counter()
    
    if plaid_index.pq_index is None or plaid_index.num_tokens == 0:
        # Empty index
        return [], [], 0.0
    
    # Embed query
    query_emb, _ = embed_tokens(
        query_text, model, tokenizer,
        max_length=max_query_length,
        normalize=True
    )
    
    query_emb = np.ascontiguousarray(query_emb.astype(np.float32))
    
    # Search for top tokens per query token
    k_tokens = min(100, plaid_index.num_tokens)
    scores, token_indices = plaid_index.pq_index.search(query_emb, k_tokens)
    
    # Aggregate by chunk
    chunk_scores = np.zeros(plaid_index.num_chunks, dtype=np.float32)
    chunk_max_contrib = np.full(
        (len(query_emb), plaid_index.num_chunks), 
        -np.inf, 
        dtype=np.float32
    )
    
    for q_idx in range(len(query_emb)):
        for t_idx, t_score in zip(token_indices[q_idx], scores[q_idx]):
            if 0 <= t_idx < len(plaid_index.token_to_chunk):
                chunk_idx = plaid_index.token_to_chunk[t_idx]
                chunk_max_contrib[q_idx, chunk_idx] = max(
                    chunk_max_contrib[q_idx, chunk_idx],
                    t_score
                )
    
    # Sum max contributions
    chunk_max_contrib = np.where(chunk_max_contrib == -np.inf, 0, chunk_max_contrib)
    chunk_scores = chunk_max_contrib.sum(axis=0)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Top-K
    top_k = min(top_k, plaid_index.num_chunks)
    top_indices = np.argsort(chunk_scores)[::-1][:top_k]
    top_scores = chunk_scores[top_indices]
    
    return top_indices.tolist(), top_scores.tolist(), latency_ms


# =============================================================================
# Index Size and Metrics
# =============================================================================

def get_plaid_index_size_mb(plaid_index: PLAIDIndex) -> float:
    """
    Calculate PLAID index size in megabytes.
    
    Components:
    - PQ codes: num_tokens × M × nbits / 8 bytes
    - Token-to-chunk mapping: num_tokens × 4 bytes
    - Chunk ranges: num_chunks × 8 bytes
    - IVF centroids: nlist × dim × 4 bytes (if IVF)
    
    Args:
        plaid_index: PLAID index
        
    Returns:
        Size in MB
    """
    # PQ codes
    pq_bytes = plaid_index.num_tokens * plaid_index.M * plaid_index.nbits // 8
    
    # Token mapping
    map_bytes = plaid_index.num_tokens * 4
    
    # Chunk ranges
    range_bytes = plaid_index.num_chunks * 8
    
    # IVF centroids (if IVF-PQ)
    ivf_bytes = 0
    if plaid_index.pq_index is not None and hasattr(plaid_index.pq_index, 'nlist'):
        ivf_bytes = plaid_index.pq_index.nlist * plaid_index.dim * 4
    
    total_bytes = pq_bytes + map_bytes + range_bytes + ivf_bytes
    return total_bytes / (1024 * 1024)


def plaid_retrieve_and_benchmark(
    query_text: str,
    chunks: List[Dict],
    plaid_index: PLAIDIndex,
    model,
    tokenizer,
    top_k: int,
    ground_truth_indices: List[int]
) -> Dict:
    """
    Retrieve with PLAID and compute benchmark metrics.
    
    Args:
        query_text: Query text
        chunks: List of chunks
        plaid_index: PLAID index
        model: BERT model
        tokenizer: BERT tokenizer
        top_k: Chunks to retrieve
        ground_truth_indices: Oracle ground-truth indices
        
    Returns:
        Dict with retrieved indices, scores, and metrics
    """
    indices, scores, latency_ms = plaid_retrieve(
        query_text, plaid_index, model, tokenizer, top_k
    )
    
    # Metrics
    gt_set = set(ground_truth_indices) if ground_truth_indices else set()
    gt_valid = len(gt_set) > 0
    
    def recall_at_k(k):
        if not gt_valid:
            return 0.0
        retrieved = set(indices[:k])
        return len(retrieved & gt_set) / len(gt_set)
    
    def mrr():
        if not gt_valid:
            return 0.0
        for rank, idx in enumerate(indices, start=1):
            if idx in gt_set:
                return 1.0 / rank
        return 0.0
    
    return {
        "retrieved_indices": indices,
        "retrieved_scores": scores,
        "recall_at_1": recall_at_k(1),
        "recall_at_5": recall_at_k(5),
        "recall_at_10": recall_at_k(10),
        "recall_at_20": recall_at_k(20),
        "mrr": mrr(),
        "latency_ms": latency_ms,
        "gt_valid": gt_valid
    }
