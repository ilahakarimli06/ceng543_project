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
    Generate summary from concatenated chunks using LED/BigBird/LongT5.
    
    Args:
        model: Loaded summarization model (LED, BigBird, or LongT5)
        tokenizer: Model tokenizer
        context_text: Concatenated chunk text
        gen_max_tokens: Maximum tokens to generate
        device: Device to run on
        
    Returns:
        Generated summary text
    """
    model.eval()
    
    # Detect model type for proper configuration
    is_led = hasattr(model.config, "model_type") and "led" in model.config.model_type.lower()
    is_bigbird = hasattr(model.config, "model_type") and "bigbird" in model.config.model_type.lower()
    is_longt5 = hasattr(model.config, "model_type") and "longt5" in model.config.model_type.lower()
    
    # Set max_length based on model
    if is_bigbird:
        max_length = 4096  # BigBird-Pegasus max input
    elif is_longt5:
        max_length = 16384  # LongT5 supports up to 16K
    else:
        max_length = 16384  # LED max input
    
    # Tokenize input
    inputs = tokenizer(
        context_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False
    ).to(device)
    
    # BigBird requires sequence length divisible by block_size (64)
    if is_bigbird:
        seq_len = inputs["input_ids"].shape[1]
        block_size = 64
        if seq_len % block_size != 0:
            padding_length = block_size - (seq_len % block_size)
            pad_ids = torch.full((1, padding_length), tokenizer.pad_token_id, dtype=torch.long, device=device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad_ids], dim=1)
            if "attention_mask" in inputs:
                pad_mask = torch.zeros((1, padding_length), dtype=torch.long, device=device)
                inputs["attention_mask"] = torch.cat([inputs["attention_mask"], pad_mask], dim=1)
    
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
