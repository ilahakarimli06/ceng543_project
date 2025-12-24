"""
ColBERT-style Token-level Late Interaction Retrieval.

Implements ColBERTv2-style retrieval with:
- Token-level embeddings from BERT
- MaxSim scoring (token-to-token similarity aggregation)
- Token → Chunk mapping for retrieval

Key insight:
- Instead of single vector per chunk (dense retrieval), each chunk has
  multiple token embeddings, enabling fine-grained matching.

Usage:
    model = load_colbert_model()
    index = build_token_index(chunks, model)
    results = colbert_retrieve(query, index, model, top_k=10)

Reference:
    Santhanam et al. "ColBERTv2: Effective and Efficient Retrieval via 
    Lightweight Late Interaction" (NAACL 2022)
"""

import torch
import numpy as np
import faiss
import time
import tempfile
import os
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ColBERTIndex:
    """ColBERT index containing token embeddings and mappings."""
    token_embeddings: np.ndarray  # (N_tokens, dim) - all token embeddings
    token_to_chunk: np.ndarray    # (N_tokens,) - token_idx → chunk_idx
    chunk_token_ranges: List[Tuple[int, int]]  # chunk_idx → (start, end) in token_embeddings
    dim: int
    num_chunks: int
    num_tokens: int


# =============================================================================
# Model Loading
# =============================================================================

_COLBERT_MODEL_CACHE = {}


def load_colbert_model(
    model_name: str = "bert-base-uncased",
    device: str = None
) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Load BERT model for token-level embeddings.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model, tokenizer)
        
    Note:
        Model is cached to avoid reloading.
    """
    global _COLBERT_MODEL_CACHE
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cache_key = f"{model_name}_{device}"
    
    if cache_key not in _COLBERT_MODEL_CACHE:
        print(f"  🔄 Loading ColBERT model: {model_name} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).eval().to(device)
        _COLBERT_MODEL_CACHE[cache_key] = (model, tokenizer)
        print(f"  ✓ ColBERT model loaded (dim={model.config.hidden_size})")
    
    return _COLBERT_MODEL_CACHE[cache_key]


def get_device(model: AutoModel) -> str:
    """Get device of model."""
    return next(model.parameters()).device


# =============================================================================
# Token Embedding
# =============================================================================

def embed_tokens(
    text: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract token-level embeddings from text.
    
    Args:
        text: Input text
        model: BERT model
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        normalize: If True, L2 normalize embeddings
        
    Returns:
        Tuple of (token_embeddings, input_ids)
        - token_embeddings: (seq_len, dim) float32 array
        - input_ids: (seq_len,) int array
    """
    device = get_device(model)
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False
    ).to(device)
    
    # Get token embeddings
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        # Use last hidden state as token embeddings
        # Shape: (1, seq_len, hidden_dim)
        token_emb = outputs.last_hidden_state.squeeze(0)  # (seq_len, dim)
    
    # Normalize if requested
    if normalize:
        token_emb = torch.nn.functional.normalize(token_emb, p=2, dim=1)
    
    # Convert to numpy
    token_embeddings = token_emb.cpu().numpy().astype(np.float32)
    input_ids = inputs['input_ids'].squeeze(0).cpu().numpy()
    
    return token_embeddings, input_ids


def embed_tokens_batch(
    texts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    batch_size: int = 32,
    normalize: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Batch embed multiple texts for efficiency.
    
    Args:
        texts: List of input texts
        model: BERT model
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length per text
        batch_size: Batch size for processing
        normalize: If True, L2 normalize embeddings
        
    Returns:
        List of (token_embeddings, input_ids) tuples
    """
    results = []
    device = get_device(model)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True  # Pad to max length in batch
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            # Shape: (batch_size, seq_len, hidden_dim)
            batch_emb = outputs.last_hidden_state
        
        # Process each item in batch
        attention_mask = inputs['attention_mask']
        input_ids = inputs['input_ids']
        
        for j in range(len(batch_texts)):
            # Get actual sequence length (excluding padding)
            seq_len = attention_mask[j].sum().item()
            
            # Extract non-padded embeddings
            emb = batch_emb[j, :seq_len, :]  # (seq_len, dim)
            ids = input_ids[j, :seq_len]      # (seq_len,)
            
            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            
            results.append((
                emb.cpu().numpy().astype(np.float32),
                ids.cpu().numpy()
            ))
    
    return results


# =============================================================================
# Index Building
# =============================================================================

def build_token_index(
    chunks: List[Dict],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    batch_size: int = 32,
    skip_special_tokens: bool = True
) -> ColBERTIndex:
    """
    Build ColBERT token-level index from chunks.
    
    Args:
        chunks: List of chunk dicts with 'text' key
        model: BERT model for token embeddings
        tokenizer: BERT tokenizer
        max_length: Maximum tokens per chunk
        batch_size: Batch size for embedding
        skip_special_tokens: If True, exclude [CLS], [SEP] from index
        
    Returns:
        ColBERTIndex with token embeddings and mappings
    """
    all_token_embeddings = []
    all_token_to_chunk = []
    chunk_token_ranges = []
    
    current_token_idx = 0
    
    # Get special token IDs to skip
    special_ids = set()
    if skip_special_tokens:
        if tokenizer.cls_token_id is not None:
            special_ids.add(tokenizer.cls_token_id)
        if tokenizer.sep_token_id is not None:
            special_ids.add(tokenizer.sep_token_id)
        if tokenizer.pad_token_id is not None:
            special_ids.add(tokenizer.pad_token_id)
    
    # Extract texts
    texts = [chunk["text"] for chunk in chunks]
    
    # Batch embed
    embeddings_list = embed_tokens_batch(
        texts, model, tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        normalize=True
    )
    
    # Build index structure
    for chunk_idx, (token_emb, input_ids) in enumerate(embeddings_list):
        chunk_start = current_token_idx
        
        for tok_idx, (emb, tok_id) in enumerate(zip(token_emb, input_ids)):
            # Skip special tokens
            if int(tok_id) in special_ids:
                continue
            
            all_token_embeddings.append(emb)
            all_token_to_chunk.append(chunk_idx)
            current_token_idx += 1
        
        chunk_end = current_token_idx
        chunk_token_ranges.append((chunk_start, chunk_end))
    
    # Stack all embeddings
    if all_token_embeddings:
        token_embeddings = np.vstack(all_token_embeddings).astype(np.float32)
    else:
        # Edge case: empty index
        dim = model.config.hidden_size
        token_embeddings = np.zeros((0, dim), dtype=np.float32)
    
    token_to_chunk = np.array(all_token_to_chunk, dtype=np.int32)
    
    return ColBERTIndex(
        token_embeddings=token_embeddings,
        token_to_chunk=token_to_chunk,
        chunk_token_ranges=chunk_token_ranges,
        dim=token_embeddings.shape[1] if token_embeddings.size > 0 else model.config.hidden_size,
        num_chunks=len(chunks),
        num_tokens=len(token_embeddings)
    )


# =============================================================================
# MaxSim Retrieval
# =============================================================================

def compute_maxsim_scores(
    query_embeddings: np.ndarray,
    index: ColBERTIndex
) -> np.ndarray:
    """
    Compute MaxSim scores for all chunks.
    
    For each query token, find max similarity with any token in each chunk.
    Sum these max similarities across query tokens to get chunk score.
    
    MaxSim(q, d) = Σ_i max_j (q_i · d_j)
    
    Args:
        query_embeddings: (query_len, dim) query token embeddings
        index: ColBERT index
        
    Returns:
        (num_chunks,) array of MaxSim scores
    """
    num_chunks = index.num_chunks
    chunk_scores = np.zeros(num_chunks, dtype=np.float32)
    
    # Fast path: use matrix operations
    if index.num_tokens > 0:
        # Compute all query-document token similarities at once
        # (query_len, num_tokens)
        all_similarities = query_embeddings @ index.token_embeddings.T
        
        # For each chunk, compute MaxSim
        for chunk_idx, (start, end) in enumerate(index.chunk_token_ranges):
            if end > start:
                # Get similarities for this chunk's tokens
                chunk_similarities = all_similarities[:, start:end]  # (query_len, chunk_tokens)
                
                # MaxSim: for each query token, take max over chunk tokens, then sum
                max_per_query = chunk_similarities.max(axis=1)  # (query_len,)
                chunk_scores[chunk_idx] = max_per_query.sum()
    
    return chunk_scores


def colbert_retrieve(
    query_text: str,
    index: ColBERTIndex,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    top_k: int = 10,
    max_query_length: int = 64
) -> Tuple[List[int], List[float], float]:
    """
    Retrieve top-K chunks using ColBERT MaxSim.
    
    Args:
        query_text: Query text
        index: ColBERT index
        model: BERT model
        tokenizer: BERT tokenizer
        top_k: Number of chunks to retrieve
        max_query_length: Maximum query tokens
        
    Returns:
        Tuple of (indices, scores, latency_ms)
    """
    # Embed query tokens
    start_time = time.perf_counter()
    
    query_emb, _ = embed_tokens(
        query_text, model, tokenizer,
        max_length=max_query_length,
        normalize=True
    )
    
    # Compute MaxSim scores
    chunk_scores = compute_maxsim_scores(query_emb, index)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Get top-K
    top_k = min(top_k, index.num_chunks)
    top_indices = np.argsort(chunk_scores)[::-1][:top_k]
    top_scores = chunk_scores[top_indices]
    
    return top_indices.tolist(), top_scores.tolist(), latency_ms


# =============================================================================
# Optimized Retrieval with FAISS
# =============================================================================

def build_faiss_token_index(
    index: ColBERTIndex,
    use_gpu: bool = False
) -> faiss.Index:
    """
    Build FAISS index over token embeddings for faster retrieval.
    
    This enables approximate nearest neighbor search for large indices.
    
    Args:
        index: ColBERT index with token embeddings
        use_gpu: If True, transfer to GPU
        
    Returns:
        FAISS IndexFlatIP over token embeddings
    """
    if index.num_tokens == 0:
        return None
    
    faiss_index = faiss.IndexFlatIP(index.dim)
    faiss_index.add(index.token_embeddings)
    
    if use_gpu and torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
        except Exception as e:
            print(f"  ⚠️ GPU FAISS failed ({e}), using CPU")
    
    return faiss_index


def colbert_retrieve_faiss(
    query_text: str,
    index: ColBERTIndex,
    faiss_index: faiss.Index,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    top_k: int = 10,
    max_query_length: int = 64,
    top_tokens_per_query: int = 100
) -> Tuple[List[int], List[float], float]:
    """
    Retrieve using FAISS for approximate token search.
    
    For each query token, find top-K similar document tokens using FAISS,
    then aggregate scores by chunk.
    
    Args:
        query_text: Query text
        index: ColBERT index
        faiss_index: FAISS index over token embeddings
        model: BERT model
        tokenizer: BERT tokenizer
        top_k: Number of chunks to retrieve
        max_query_length: Maximum query tokens
        top_tokens_per_query: Number of document tokens to retrieve per query token
        
    Returns:
        Tuple of (indices, scores, latency_ms)
    """
    start_time = time.perf_counter()
    
    # Embed query tokens
    query_emb, _ = embed_tokens(
        query_text, model, tokenizer,
        max_length=max_query_length,
        normalize=True
    )
    
    # Ensure contiguous
    query_emb = np.ascontiguousarray(query_emb.astype(np.float32))
    
    # Use FAISS to find top tokens for each query token
    k_tokens = min(top_tokens_per_query, index.num_tokens)
    scores, token_indices = faiss_index.search(query_emb, k_tokens)
    
    # Aggregate by chunk (approximate MaxSim)
    chunk_scores = np.zeros(index.num_chunks, dtype=np.float32)
    chunk_max_contrib = np.full((len(query_emb), index.num_chunks), -np.inf, dtype=np.float32)
    
    for q_idx in range(len(query_emb)):
        for t_idx, t_score in zip(token_indices[q_idx], scores[q_idx]):
            if t_idx >= 0 and t_idx < len(index.token_to_chunk):
                chunk_idx = index.token_to_chunk[t_idx]
                chunk_max_contrib[q_idx, chunk_idx] = max(
                    chunk_max_contrib[q_idx, chunk_idx],
                    t_score
                )
    
    # Sum max contributions per query token
    # Replace -inf with 0 for chunks not hit
    chunk_max_contrib = np.where(chunk_max_contrib == -np.inf, 0, chunk_max_contrib)
    chunk_scores = chunk_max_contrib.sum(axis=0)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Get top-K
    top_k = min(top_k, index.num_chunks)
    top_indices = np.argsort(chunk_scores)[::-1][:top_k]
    top_scores = chunk_scores[top_indices]
    
    return top_indices.tolist(), top_scores.tolist(), latency_ms


# =============================================================================
# Index Size and Metrics
# =============================================================================

def get_colbert_index_size_mb(index: ColBERTIndex) -> float:
    """
    Calculate ColBERT index size in megabytes.
    
    Components:
    - Token embeddings: num_tokens × dim × 4 bytes
    - Token-to-chunk mapping: num_tokens × 4 bytes
    - Chunk ranges: num_chunks × 8 bytes (2 ints)
    
    Args:
        index: ColBERT index
        
    Returns:
        Size in MB
    """
    # Token embeddings (float32)
    emb_bytes = index.token_embeddings.nbytes
    
    # Token-to-chunk mapping (int32)
    map_bytes = index.token_to_chunk.nbytes
    
    # Chunk ranges (2 ints per chunk)
    range_bytes = index.num_chunks * 8
    
    total_bytes = emb_bytes + map_bytes + range_bytes
    return total_bytes / (1024 * 1024)


def colbert_retrieve_and_benchmark(
    query_text: str,
    chunks: List[Dict],
    index: ColBERTIndex,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    top_k: int,
    ground_truth_indices: List[int],
    use_faiss: bool = False,
    faiss_index: Optional[faiss.Index] = None
) -> Dict:
    """
    Retrieve top-K chunks and compute benchmark metrics.
    
    Args:
        query_text: Query text
        chunks: List of chunk dicts (order must match index)
        index: ColBERT index
        model: BERT model
        tokenizer: BERT tokenizer
        top_k: Number of chunks to retrieve
        ground_truth_indices: Oracle ground-truth chunk indices
        use_faiss: If True, use FAISS for approximate search
        faiss_index: Pre-built FAISS index (required if use_faiss=True)
        
    Returns:
        Dict with retrieved indices, scores, and metrics
    """
    if use_faiss and faiss_index is not None:
        indices, scores, latency_ms = colbert_retrieve_faiss(
            query_text, index, faiss_index, model, tokenizer, top_k
        )
    else:
        indices, scores, latency_ms = colbert_retrieve(
            query_text, index, model, tokenizer, top_k
        )
    
    # Compute metrics
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
