"""
Chunking strategies for document segmentation.

Supports three methods:
1. fixed: Token-based sliding window
2. sentence: Sentence-aligned boundaries
3. semantic: Embedding similarity-based topic segmentation

Output format: List[dict] with {"text": str, "ids": List[int], "start_token": int, "end_token": int}
"""

from transformers import AutoTokenizer
import nltk
from typing import List, Dict, Union
import numpy as np

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Global cache for embedding model (avoid reloading)
_EMBEDDING_MODEL_CACHE = {}


def _normalize_overlap(overlap: Union[int, float], chunk_size: int) -> int:
    """
    Normalize overlap parameter to token count.
    
    Args:
        overlap: Either token count (int) or ratio (float between 0 and 1)
        chunk_size: Chunk size in tokens
        
    Returns:
        Overlap in tokens
    """
    if isinstance(overlap, float):
        if not (0.0 <= overlap <= 1.0):
            raise ValueError(f"Overlap ratio must be between 0 and 1, got {overlap}")
        return int(chunk_size * overlap)
    elif isinstance(overlap, int):
        if overlap < 0:
            raise ValueError(f"Overlap must be non-negative, got {overlap}")
        return overlap
    else:
        raise TypeError(f"Overlap must be int or float, got {type(overlap)}")


def chunk_text_fixed(
    text: str,
    tokenizer,
    chunk_size: int,
    overlap: Union[int, float]
) -> List[Dict]:
    """
    Fixed-size token-based chunking with sliding window.
    
    Args:
        text: Input document text
        tokenizer: HuggingFace tokenizer
        chunk_size: Number of tokens per chunk
        overlap: Number of overlapping tokens (int) or ratio (float 0-1)
        
    Returns:
        List of chunk dicts with keys: text, ids, start_token, end_token
    """
    # Normalize overlap
    overlap_tokens = _normalize_overlap(overlap, chunk_size)
    
    # Tokenize using modern API
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    
    # Calculate step size
    step = max(1, chunk_size - overlap_tokens)
    
    chunks = []
    for i in range(0, len(token_ids), step):
        chunk_ids = token_ids[i:i + chunk_size]
        if not chunk_ids:
            break
        
        # Decode to text
        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        chunks.append({
            "text": chunk_text.strip(),
            "ids": chunk_ids,
            "start_token": i,
            "end_token": i + len(chunk_ids)
        })
        
        # Stop if we've covered the entire document
        if i + chunk_size >= len(token_ids):
            break
    
    return chunks


def chunk_text_sentence(
    text: str,
    tokenizer,
    chunk_size: int,
    overlap: Union[int, float]
) -> List[Dict]:
    """
    Sentence-aligned chunking: group sentences to approximate chunk_size.
    
    Args:
        text: Input document text
        tokenizer: HuggingFace tokenizer
        chunk_size: Target number of tokens per chunk
        overlap: Number of overlapping tokens (int) or ratio (float 0-1)
        
    Returns:
        List of chunk dicts with keys: text, ids, start_token, end_token
    """
    # Normalize overlap
    overlap_tokens = _normalize_overlap(overlap, chunk_size)
    
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Pre-compute token IDs and counts for all sentences (performance optimization)
    sentence_data = []
    for sent in sentences:
        sent_ids = tokenizer(sent, add_special_tokens=False)["input_ids"]
        sentence_data.append({
            "text": sent,
            "ids": sent_ids,
            "token_count": len(sent_ids)
        })
    
    chunks = []
    current_sentences = []
    current_ids = []
    current_token_count = 0
    global_token_offset = 0
    
    for sent_data in sentence_data:
        sent_token_count = sent_data["token_count"]
        
        # Handle oversized single sentence (edge case)
        if sent_token_count > chunk_size:
            # Finalize current chunk if exists
            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    "text": chunk_text.strip(),
                    "ids": current_ids.copy(),
                    "start_token": global_token_offset - len(current_ids),
                    "end_token": global_token_offset
                })
                current_sentences = []
                current_ids = []
                current_token_count = 0
            
            # Split oversized sentence using fixed chunking
            oversized_chunks = chunk_text_fixed(
                sent_data["text"],
                tokenizer,
                chunk_size,
                overlap_tokens
            )
            for chunk in oversized_chunks:
                chunk["start_token"] += global_token_offset
                chunk["end_token"] += global_token_offset
                chunks.append(chunk)
            
            global_token_offset += sent_token_count
            continue
        
        # If adding this sentence exceeds chunk_size, finalize current chunk
        if current_token_count + sent_token_count > chunk_size and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "text": chunk_text.strip(),
                "ids": current_ids.copy(),
                "start_token": global_token_offset - len(current_ids),
                "end_token": global_token_offset
            })
            
            # Handle overlap: keep last few sentences for next chunk
            if overlap_tokens > 0:
                overlap_sentences = []
                overlap_ids = []
                overlap_token_count = 0
                
                for sent, sent_ids, sent_count in zip(
                    reversed(current_sentences),
                    reversed([s["ids"] for s in sentence_data[len(sentence_data) - len(current_sentences):]]),
                    reversed([s["token_count"] for s in sentence_data[len(sentence_data) - len(current_sentences):]])
                ):
                    if overlap_token_count + sent_count <= overlap_tokens:
                        overlap_sentences.insert(0, sent)
                        overlap_ids = sent_ids + overlap_ids
                        overlap_token_count += sent_count
                    else:
                        break
                
                current_sentences = overlap_sentences
                current_ids = overlap_ids
                current_token_count = overlap_token_count
                global_token_offset -= overlap_token_count
            else:
                current_sentences = []
                current_ids = []
                current_token_count = 0
        
        # Add current sentence
        current_sentences.append(sent_data["text"])
        current_ids.extend(sent_data["ids"])
        current_token_count += sent_token_count
        global_token_offset += sent_token_count
    
    # Add final chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append({
            "text": chunk_text.strip(),
            "ids": current_ids.copy(),
            "start_token": global_token_offset - len(current_ids),
            "end_token": global_token_offset
        })
    
    return chunks


def chunk_text_semantic(
    text: str,
    tokenizer,
    chunk_size: int,
    overlap: Union[int, float],
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold: float = 0.7
) -> List[Dict]:
    """
    Semantic chunking: detect topic shifts using embedding similarity.
    
    Args:
        text: Input document text
        tokenizer: HuggingFace tokenizer
        chunk_size: Target number of tokens per chunk
        overlap: Number of overlapping tokens (int) or ratio (float 0-1)
        embedding_model_name: Sentence-transformers model for embeddings
        similarity_threshold: Cosine similarity threshold for topic shift detection
        
    Returns:
        List of chunk dicts with keys: text, ids, start_token, end_token
    """
    # Normalize overlap
    overlap_tokens = _normalize_overlap(overlap, chunk_size)
    
    # Import here to avoid circular dependency
    from sentence_transformers import SentenceTransformer
    
    # Load embedding model with caching (performance optimization)
    if embedding_model_name not in _EMBEDDING_MODEL_CACHE:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _EMBEDDING_MODEL_CACHE[embedding_model_name] = SentenceTransformer(
            embedding_model_name,
            device=device
        )
    embedding_model = _EMBEDDING_MODEL_CACHE[embedding_model_name]
    
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    
    if len(sentences) <= 1:
        # Fallback to fixed chunking if too few sentences
        return chunk_text_fixed(text, tokenizer, chunk_size, overlap_tokens)
    
    # Embed all sentences (batch processing for performance)
    sentence_embeddings = embedding_model.encode(
        sentences,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True  # L2 normalization
    )
    
    # Calculate cosine similarity between consecutive sentences
    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        emb1 = sentence_embeddings[i]
        emb2 = sentence_embeddings[i + 1]
        # Dot product = cosine similarity (since normalized)
        sim = np.dot(emb1, emb2)
        similarities.append(sim)
    
    # Detect topic boundaries (low similarity = topic shift)
    topic_boundaries = [0]  # Start of document
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            topic_boundaries.append(i + 1)
    topic_boundaries.append(len(sentences))  # End of document
    
    # Pre-compute sentence token data
    sentence_data = []
    for sent in sentences:
        sent_ids = tokenizer(sent, add_special_tokens=False)["input_ids"]
        sentence_data.append({
            "text": sent,
            "ids": sent_ids,
            "token_count": len(sent_ids)
        })
    
    # Group sentences into chunks based on topic boundaries
    chunks = []
    global_token_offset = 0
    
    for i in range(len(topic_boundaries) - 1):
        start_idx = topic_boundaries[i]
        end_idx = topic_boundaries[i + 1]
        
        topic_sentences = sentence_data[start_idx:end_idx]
        topic_text = " ".join([s["text"] for s in topic_sentences])
        topic_ids = []
        for s in topic_sentences:
            topic_ids.extend(s["ids"])
        topic_token_count = len(topic_ids)
        
        # If topic segment is too large, split it using sentence-aligned chunking
        if topic_token_count > chunk_size * 1.5:
            # Split large topic into smaller chunks
            sub_chunks = chunk_text_sentence(topic_text, tokenizer, chunk_size, overlap_tokens)
            for chunk in sub_chunks:
                chunk["start_token"] += global_token_offset
                chunk["end_token"] += global_token_offset
                chunks.append(chunk)
        else:
            chunks.append({
                "text": topic_text.strip(),
                "ids": topic_ids,
                "start_token": global_token_offset,
                "end_token": global_token_offset + topic_token_count
            })
        
        global_token_offset += topic_token_count
    
    return chunks


def chunk_text(
    text: str,
    tokenizer,
    chunk_size: int,
    overlap: Union[int, float],
    method: str = "fixed",
    **kwargs
) -> List[Dict]:
    """
    Main chunking function with method selection.
    
    Args:
        text: Input document text
        tokenizer: HuggingFace tokenizer
        chunk_size: Number of tokens per chunk
        overlap: Number of overlapping tokens (int) or ratio (float 0-1)
        method: Chunking method ("fixed", "sentence", "semantic")
        **kwargs: Additional arguments for specific methods
            - embedding_model_name: For semantic method
            - similarity_threshold: For semantic method
        
    Returns:
        List of chunk dicts with keys: text, ids, start_token, end_token
    """
    if method == "fixed":
        return chunk_text_fixed(text, tokenizer, chunk_size, overlap)
    elif method == "sentence":
        return chunk_text_sentence(text, tokenizer, chunk_size, overlap)
    elif method == "semantic":
        return chunk_text_semantic(text, tokenizer, chunk_size, overlap, **kwargs)
    else:
        raise ValueError(f"Unknown chunking method: {method}. Use 'fixed', 'sentence', or 'semantic'.")
