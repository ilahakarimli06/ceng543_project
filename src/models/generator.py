"""
Generation module for chunk retrieval pipeline.

Handles:
- Concatenating retrieved chunks with token budget enforcement
- Summary generation using LED/LongT5 models
"""

from typing import List, Dict
import torch


def concatenate_chunks(
    chunks: List[Dict],
    chunk_indices: List[int],
    tokenizer,
    max_tokens: int = 16000,
    preserve_order: bool = False
) -> str:
    """
    Concatenate top-K retrieved chunks respecting token budget.
    
    Args:
        chunks: Full list of chunk dicts with keys: text, ids, start_token, end_token
        chunk_indices: Indices of retrieved chunks (in relevance order)
        tokenizer: HuggingFace tokenizer
        max_tokens: Maximum total tokens (default: 16k for LED)
        preserve_order: If True, reorder chunks by document position
        
    Returns:
        Concatenated text within token budget
    """
    # Get retrieved chunks
    retrieved_chunks = [chunks[i] for i in chunk_indices]
    
    # Optionally reorder by document position (for coherence)
    if preserve_order:
        # Sort by original start_token position
        sorted_pairs = sorted(zip(chunk_indices, retrieved_chunks), key=lambda x: x[1]["start_token"])
        retrieved_chunks = [chunk for _, chunk in sorted_pairs]
    
    # Concatenate chunks while respecting token budget
    combined_text = ""
    current_tokens = 0
    
    for chunk in retrieved_chunks:
        # Use pre-computed token IDs from chunk dict (avoid re-tokenization)
        chunk_token_count = len(chunk["ids"])
        
        # Check if adding this chunk exceeds budget
        if current_tokens + chunk_token_count > max_tokens:
            # Try to add partial chunk if there's room
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 50:  # Only add if meaningful amount remains
                partial_ids = chunk["ids"][:remaining_tokens]
                partial_text = tokenizer.decode(partial_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                combined_text += " " + partial_text.strip()
            break
        
        # Add full chunk text
        combined_text += " " + chunk["text"].strip()
        current_tokens += chunk_token_count
    
    return combined_text.strip()


def generate_from_chunks(
    model,
    tokenizer,
    context_text: str,
    gen_max_tokens: int = 256,
    device: str = "cuda"
) -> str:
    """
    Generate summary from concatenated chunks using LED/LongT5.
    
    Args:
        model: Loaded summarization model (LED or LongT5)
        tokenizer: Model tokenizer
        context_text: Concatenated chunk text
        gen_max_tokens: Maximum tokens to generate
        device: Device to run on
        
    Returns:
        Generated summary text
    """
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        context_text,
        return_tensors="pt",
        truncation=True,
        max_length=16384,  # LED max input
        padding=False
    ).to(device)
    
    # Check if LED model (for global attention)
    is_led = hasattr(model.config, "model_type") and "led" in model.config.model_type.lower()
    
    if is_led:
        # Add global attention to first token (BOS)
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[:, 0] = 1  # First token gets global attention
        inputs["global_attention_mask"] = global_attention_mask
    
    # Generate summary
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=gen_max_tokens,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode
    summary = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return summary.strip()
