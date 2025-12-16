"""
Dense retrieval module using sentence-transformers and FAISS.

Supports:
- Embedding generation with MiniLM/BGE models
- FAISS IndexFlatIP for dot-product similarity
- Top-K chunk retrieval

IMPORTANT: Load embedding model ONCE and pass as parameter to avoid reloading overhead.
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple, Dict, Optional
import torch
import os
import hashlib


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load sentence-transformers embedding model.
    
    Args:
        model_name: Model identifier (MiniLM or BGE)
        
    Returns:
        SentenceTransformer model
        
    Note:
        Call this ONCE at the start and pass the model to embed_chunks() and retrieve_top_k()
        to avoid reloading overhead.
    """
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model


def embed_chunks(
    chunks: List[Dict],
    embedding_model: SentenceTransformer
) -> np.ndarray:
    """
    Embed list of chunk dicts using pre-loaded sentence-transformers model.
    
    Args:
        chunks: List of chunk dicts with keys: text, ids, start_token, end_token
        embedding_model: Pre-loaded SentenceTransformer model
        
    Returns:
        Numpy array of embeddings (shape: [num_chunks, embedding_dim])
        Contiguous float32 array ready for FAISS
    """
    # Extract text from chunk dicts
    chunk_texts = [chunk["text"] for chunk in chunks]
    
    # Encode chunks to embeddings
    embeddings = embedding_model.encode(
        chunk_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True  # L2 normalization for cosine similarity
    )
    
    # Ensure contiguous float32 array for FAISS compatibility
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    
    return embeddings


def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    """
    Build FAISS IndexFlatIP (inner product) for dense retrieval.
    
    Args:
        embeddings: Numpy array of embeddings (shape: [num_chunks, embedding_dim])
                   Should be contiguous float32 array
        use_gpu: If True, transfer index to GPU (requires faiss-gpu)
        
    Returns:
        FAISS index
    """
    embedding_dim = embeddings.shape[1]
    
    # Ensure contiguous float32 (defensive check)
    if not embeddings.flags['C_CONTIGUOUS']:
        embeddings = np.ascontiguousarray(embeddings)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    # Create IndexFlatIP (dot product similarity)
    # Since embeddings are L2-normalized, dot product = cosine similarity
    index = faiss.IndexFlatIP(embedding_dim)
    
    # Transfer to GPU if requested
    if use_gpu and torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            print(f"⚠️ GPU FAISS failed ({e}), using CPU")
    
    # Add embeddings to index
    index.add(embeddings)
    
    return index


def retrieve_top_k(
    query_text: str,
    chunks: List[Dict],
    index: faiss.Index,
    K: int,
    embedding_model: SentenceTransformer
) -> Tuple[List[int], List[float]]:
    """
    Retrieve top-K most relevant chunks for a query.
    
    Args:
        query_text: Query text (e.g., document title or first N tokens)
        chunks: List of chunk dicts (used for validation, not in search)
        index: FAISS index
        K: Number of chunks to retrieve
        embedding_model: Pre-loaded SentenceTransformer model (MUST be same as used for chunks)
        
    Returns:
        Tuple of (top_k_indices, top_k_scores)
        
    Note:
        - Embedding model MUST be the same one used to create the index
        - Handles FAISS edge cases (e.g., -1 indices when K > num_chunks)
    """
    # Embed query using the SAME model as chunks
    query_embedding = embedding_model.encode(
        [query_text],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    
    # Ensure contiguous float32
    query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
    
    # Clamp K to number of chunks (FAISS edge case)
    num_chunks = len(chunks)
    K_clamped = min(K, num_chunks)
    
    # Search FAISS index
    scores, indices = index.search(query_embedding, K_clamped)
    
    # Convert to lists and filter out invalid indices (-1)
    top_k_indices = []
    top_k_scores = []
    for idx, score in zip(indices[0], scores[0]):
        if idx >= 0 and idx < num_chunks:  # Valid index
            top_k_indices.append(int(idx))
            top_k_scores.append(float(score))
    
    return top_k_indices, top_k_scores


def create_query_from_document(
    text: str,
    tokenizer,
    strategy: str = "first_tokens",
    num_tokens: int = 128
) -> str:
    """
    Create query representation from document.
    
    Args:
        text: Full document text
        tokenizer: HuggingFace tokenizer
        strategy: Query strategy ("first_tokens", "title", or "hybrid")
        num_tokens: Number of tokens to use for "first_tokens" strategy
        
    Returns:
        Query text
    """
    if strategy == "first_tokens":
        # Use first N tokens as query (modern tokenizer API)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        query_ids = token_ids[:num_tokens]
        query_text = tokenizer.decode(query_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return query_text.strip()
    
    elif strategy == "title":
        # Extract title (assume first line or first sentence)
        lines = text.split('\n')
        if lines:
            return lines[0].strip()
        else:
            # Fallback to first sentence
            import nltk
            sentences = nltk.sent_tokenize(text)
            return sentences[0] if sentences else text[:200]
    
    elif strategy == "hybrid":
        # Combine title + first N tokens
        title = create_query_from_document(text, tokenizer, strategy="title")
        first_tokens = create_query_from_document(text, tokenizer, strategy="first_tokens", num_tokens=num_tokens)
        return f"{title} {first_tokens}".strip()
    
    else:
        raise ValueError(f"Unknown query strategy: {strategy}")


def _compute_cache_key(chunks: List[Dict], embedding_model_name: str) -> str:
    """
    Compute a unique cache key for chunks and embedding model.
    
    Args:
        chunks: List of chunk dicts
        embedding_model_name: Embedding model identifier
        
    Returns:
        MD5 hash as cache key
    """
    # Create a deterministic string from chunks
    chunk_texts = "|".join([chunk["text"] for chunk in chunks])
    cache_string = f"{embedding_model_name}::{chunk_texts}"
    
    # Compute MD5 hash
    cache_key = hashlib.md5(cache_string.encode('utf-8')).hexdigest()
    return cache_key


def save_embeddings(embeddings: np.ndarray, cache_dir: str, cache_key: str) -> str:
    """
    Save embeddings to disk for reproducibility.
    
    Args:
        embeddings: Numpy array of embeddings
        cache_dir: Directory to save cache files
        cache_key: Unique identifier for this embedding set
        
    Returns:
        Path to saved file
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"embeddings_{cache_key}.npy")
    np.save(cache_path, embeddings)
    return cache_path


def load_embeddings(cache_dir: str, cache_key: str) -> Optional[np.ndarray]:
    """
    Load embeddings from disk cache.
    
    Args:
        cache_dir: Directory containing cache files
        cache_key: Unique identifier for this embedding set
        
    Returns:
        Numpy array of embeddings, or None if not found
    """
    cache_path = os.path.join(cache_dir, f"embeddings_{cache_key}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    return None


def save_index(index: faiss.Index, cache_dir: str, cache_key: str) -> str:
    """
    Save FAISS index to disk for reproducibility.
    
    Args:
        index: FAISS index
        cache_dir: Directory to save cache files
        cache_key: Unique identifier for this index
        
    Returns:
        Path to saved file
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"index_{cache_key}.faiss")
    faiss.write_index(index, cache_path)
    return cache_path


def load_index(cache_dir: str, cache_key: str) -> Optional[faiss.Index]:
    """
    Load FAISS index from disk cache.
    
    Args:
        cache_dir: Directory containing cache files
        cache_key: Unique identifier for this index
        
    Returns:
        FAISS index, or None if not found
    """
    cache_path = os.path.join(cache_dir, f"index_{cache_key}.faiss")
    if os.path.exists(cache_path):
        return faiss.read_index(cache_path)
    return None
