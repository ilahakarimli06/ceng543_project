from transformers import LEDForConditionalGeneration, BigBirdPegasusForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_model(name, attention_impl="default", device="cuda", use_bf16=True, use_compile=True):
    """
    Load encoder-decoder model with H100 optimizations.
    
    Supports:
        - LED (BART-based): allenai/led-base-16384
        - BigBird-Pegasus: google/bigbird-pegasus-large-arxiv
        - LongT5 (T5-based): google/long-t5-tglobal-base
    
    Args:
        attention_impl: "default" (standard PyTorch) or "flash_attention_2" (optimized)
        use_bf16: Use bfloat16 mixed precision (optimal for H100)
        use_compile: Use torch.compile for H100 optimization
    
    Returns:
        model, tokenizer, metadata dict (for reproducibility logging)
    """
    metadata = {
        "attention_impl": attention_impl,
        "attention_backend": "unknown",
        "flash_attn_version": None,
        "model_family": "unknown",
        "dtype": "bfloat16" if use_bf16 else "float32",
        "torch_compile": use_compile,
    }
    
    # Detect model type
    name_lower = name.lower()
    is_led = "led" in name_lower
    is_bigbird = "bigbird" in name_lower
    is_longt5 = "long-t5" in name_lower or "longt5" in name_lower
    
    # H100 optimization: Use BF16 for faster inference
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    if is_led:
        metadata["model_family"] = "LED"
        
        # Load LED with standard attention (BF16)
        # Flash Attention can be applied via runtime patching (see flash_led_patch.py)
        model = LEDForConditionalGeneration.from_pretrained(name, dtype=dtype)
        
        # Apply Flash Attention patch if requested
        # MUST happen before torch.compile
        if attention_impl == "flash_attention_2":
            try:
                from src.models.flash_led_patch import patch_led_model
                print("🔧 Applying Flash Attention patch to LED model...")
                patched = patch_led_model(model)
                if patched > 0:
                    print(f"✓ Patched {patched} attention modules with Flash Attention")
                    metadata["attention_backend"] = "flash_attention_2_patched"
                    try:
                        import flash_attn
                        metadata["flash_attn_version"] = flash_attn.__version__
                    except:
                        metadata["flash_attn_version"] = "unknown"
                else:
                    print("⚠ Flash Attention patch failed - using standard attention")
                    metadata["attention_impl"] = "default"
            except Exception as e:
                print(f"⚠ Flash Attention patch failed ({e}) - using standard attention")
                metadata["attention_impl"] = "default"
        else:
            metadata["attention_backend"] = "eager"
            metadata["attention_impl"] = "default"
        
        tok = AutoTokenizer.from_pretrained(name)
        
    elif is_bigbird:
        metadata["model_family"] = "BigBird-Pegasus"
        # BigBird uses block_sparse attention for efficient long-sequence processing
        model = BigBirdPegasusForConditionalGeneration.from_pretrained(
            name,
            attention_type="block_sparse",  # Efficient for long sequences
            dtype=dtype  # Use dtype instead of torch_dtype (new API)
        )
        metadata["attention_backend"] = "block_sparse"  # BigBird's built-in block_sparse attention
        metadata["attention_type"] = "block_sparse"
        metadata["max_length"] = 4096  # BigBird-Pegasus max length
        
        # Get block size if available
        if hasattr(model.config, 'block_size'):
            metadata["block_size"] = model.config.block_size
        
        tok = AutoTokenizer.from_pretrained(name)
        print(f"✓ Using BigBird-Pegasus with block_sparse attention")
        
    elif is_longt5:
        metadata["model_family"] = "LongT5"
        # LongT5 with transient global attention - optimized for summarization
        model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=dtype)
        
        # Detect attention type from model name
        if "tglobal" in name_lower:
            metadata["attention_backend"] = "transient_global"
            metadata["attention_type"] = "transient_global"
        elif "local" in name_lower:
            metadata["attention_backend"] = "local"
            metadata["attention_type"] = "local"
        else:
            metadata["attention_backend"] = "default"
        
        metadata["max_length"] = 16384  # LongT5 supports up to 16K tokens
        
        tok = AutoTokenizer.from_pretrained(name)
        print(f"✓ Using LongT5 with {metadata['attention_backend']} attention")
        
    else:
        raise ValueError(f"Unsupported model: {name}. Use LED, BigBird, or LongT5 models.")

    # PAD token setup
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    
    # Sync tokenizer and model config
    model.config.pad_token_id = tok.pad_token_id    
    
    # Set decoder_start_token_id (both LED and LongT5 use pad_token_id)
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tok.pad_token_id
    
    # Extract model metadata for reproducibility
    metadata["model_commit_hash"] = getattr(model.config, "_commit_hash", "unknown")
    
    # Get transformers version
    import transformers
    metadata["tokenizer_version"] = transformers.__version__
    
    # Move to device
    model = model.to(device)
    
    # H100 optimization: Use torch.compile for faster inference
    if use_compile:
        try:
            print("🚀 Compiling model with torch.compile for H100...")
            model = torch.compile(model, mode="reduce-overhead")
            metadata["torch_compile"] = True
            print("✓ Model compiled successfully")
        except Exception as e:
            print(f"⚠ torch.compile failed ({e}), continuing without compilation")
            metadata["torch_compile"] = False
    
    return model, tok, metadata

def generate_with_windows(model, tokenizer, windows, gen_max=256, global_tokens=0, device="cuda", batch_size=32, aggregation="concat"):
    """
    Process windows in batches for better H100 GPU utilization.
    Token-ID seviyesinde çalışır - decode/re-tokenize akışını kullanmaz.
    
    Args:
        global_tokens: LED için ilk N token'a global attention ver (0, 16, 64)
        batch_size: Number of windows to process simultaneously (H100 optimization)
        aggregation: "concat" (default, just join summaries) or "hierarchical" (summarize the summaries)
    """
    # Route BigBird models to dedicated generation function
    is_bigbird = hasattr(model.config, "model_type") and "bigbird" in model.config.model_type.lower()
    if is_bigbird:
        return generate_with_windows_bigbird(model, tokenizer, windows, gen_max, device, batch_size, aggregation)
    model.eval()
    outputs = []

    # LED kapasitesi belirleme
    led_cap = 16384 if getattr(model.config, "model_type", "") == "led" else None
    cfg_cap = getattr(model.config, "max_position_embeddings", None)
    tok_cap = getattr(tokenizer, "model_max_length", None)
    cap = next((c for c in [led_cap, cfg_cap, tok_cap, 4096] if isinstance(c, int) and c > 0), 4096)

    # LED encoder için special token ekleme
    is_led = getattr(model.config, "model_type", "") == "led"
    
    # Process windows in batches for H100 optimization
    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start:batch_start + batch_size]
        batch_input_ids = []
        batch_attention_masks = []
        batch_global_masks = []
        
        for window_ids in batch_windows:
            # Token ID'leri üzerinde doğrudan çalış
            if is_led:
                chunk = window_ids[:cap]  # LED: sadece kapasite kontrolü
            else:
                # Diğer modeller için BOS/EOS ekle
                bos_id = tokenizer.bos_token_id
                eos_id = tokenizer.eos_token_id
                chunk = window_ids[:cap-2]  # BOS/EOS için yer bırak
                if bos_id is not None:
                    chunk = [bos_id] + chunk
                if eos_id is not None:
                    chunk = chunk + [eos_id]
            
            batch_input_ids.append(chunk)
        
        # Pad sequences to same length in batch
        max_len = max(len(seq) for seq in batch_input_ids)
        padded_input_ids = []
        attention_masks = []
        
        for chunk in batch_input_ids:
            # Pad to max_len
            padding_length = max_len - len(chunk)
            padded_chunk = chunk + [tokenizer.pad_token_id] * padding_length
            padded_input_ids.append(padded_chunk)
            
            # Attention mask: 1 for real tokens, 0 for padding
            attention_mask = [1] * len(chunk) + [0] * padding_length
            attention_masks.append(attention_mask)
        
        # Convert to tensors
        input_ids = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
        
        # Input dict oluştur
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # LED ise global attention mask ayarla (SMART PLACEMENT)
        is_led_model = hasattr(model.config, "model_type") and "led" in model.config.model_type.lower()
        
        if is_led_model and global_tokens > 0:
            global_attention_mask = torch.zeros_like(input_ids)
            
            for batch_idx, chunk in enumerate(batch_input_ids):
                seq_len = len(chunk)
                n_global = min(global_tokens, seq_len)
                
                # Task-aware global attention placement for summarization
                global_positions = []
                
                # 1. Always include BOS token (first token)
                global_positions.append(0)
                
                # 2. Find sentence boundaries (periods, newlines) for remaining slots
                if n_global > 1:
                    # Decode to find sentence markers
                    text_tokens = tokenizer.convert_ids_to_tokens(chunk)
                    
                    # Find positions of sentence boundaries
                    boundary_positions = []
                    for i, token in enumerate(text_tokens):
                        # Look for period, newline, or section markers
                        if any(marker in str(token).lower() for marker in ['.', '\n', '</s>', 'Ċ', 'Ġ.']):
                            boundary_positions.append(i)
                    
                    # Select evenly distributed boundaries
                    if boundary_positions:
                        # Take every Nth boundary to distribute global attention
                        step = max(1, len(boundary_positions) // (n_global - 1))
                        selected = boundary_positions[::step][:n_global - 1]
                        global_positions.extend(selected)
                
                # 3. If still need more, distribute evenly across sequence
                while len(global_positions) < n_global:
                    # Add evenly spaced positions
                    spacing = seq_len // (n_global - len(global_positions) + 1)
                    next_pos = spacing * (len(global_positions))
                    if next_pos < seq_len and next_pos not in global_positions:
                        global_positions.append(next_pos)
                    else:
                        break
                
                # Apply global attention to selected positions
                for pos in global_positions[:n_global]:
                    if pos < max_len:
                        global_attention_mask[batch_idx, pos] = 1
            
            inputs["global_attention_mask"] = global_attention_mask

        # Generate for entire batch
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=gen_max,
                num_beams=4,
                early_stopping=True
            )

        # Decode batch outputs
        for i in range(len(batch_windows)):
            summary = tokenizer.decode(
                out_ids[i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            outputs.append(summary.strip())

    # Aggregation strategy
    if aggregation == "hierarchical" and len(outputs) > 1:
        # Hierarchical: summarize the concatenated summaries
        combined_text = " ".join(outputs)
        
        # Tokenize the combined summaries
        combined_ids = tokenizer.encode(combined_text, add_special_tokens=False)
        
        # Check if it fits in model capacity
        is_led_model = hasattr(model.config, "model_type") and "led" in model.config.model_type.lower()
        cap = 16384 if is_led_model else 4096
        
        # Truncate if needed
        combined_ids = combined_ids[:cap]
        
        # Create input tensors
        input_ids = torch.tensor([combined_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        
        # Add global attention for LED
        if is_led_model and global_tokens > 0:
            global_attention_mask = torch.zeros_like(input_ids)
            n_global = min(global_tokens, len(combined_ids))
            global_attention_mask[0, :n_global] = 1
            inputs["global_attention_mask"] = global_attention_mask
        
        # Generate final summary
        with torch.no_grad():
            final_ids = model.generate(
                **inputs,
                max_new_tokens=gen_max,
                num_beams=4,
                early_stopping=True
            )
        
        return tokenizer.decode(final_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    
    # Default: concat (original behavior)
    return " ".join(outputs)

def generate_with_windows_bigbird(model, tokenizer, windows, gen_max=256, device="cuda", batch_size=32, aggregation="concat"):
    """
    Generate summaries with BigBird-Pegasus using batch processing for H100 optimization.
    BigBird-Pegasus max input: 4096 tokens
    
    Key differences from LED:
    - Max input length: 4096 (vs 16384 for LED)
    - Block size alignment: Sequences must be divisible by block_size (64)
    - No global_attention_mask needed (block_sparse attention is automatic)
    - Batch processing for H100 GPU utilization
    
    Args:
        model: BigBird-Pegasus model
        tokenizer: BigBird tokenizer
        windows: List of token ID lists (windows)
        gen_max: Max tokens to generate per window
        device: Device to run on
        batch_size: Number of windows to process simultaneously (H100 optimization)
        aggregation: "concat" (default) or "hierarchical" (summarize the summaries)
    """
    cap = 4096  # BigBird-Pegasus max input length
    summaries = []
    
    # Process windows in batches for H100 optimization
    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start:batch_start + batch_size]
        batch_chunks = []
        
        for window_ids in batch_windows:
            # Truncate to max length
            chunk = window_ids[:cap]
            
            # Ensure length is divisible by block_size (64)
            block_size = 64
            if len(chunk) % block_size != 0:
                # Pad to nearest multiple of block_size with pad_token_id
                padding_length = block_size - (len(chunk) % block_size)
                chunk = chunk + [tokenizer.pad_token_id] * padding_length
            
            batch_chunks.append(chunk)
        
        # Pad all sequences to same length in batch
        max_len = max(len(c) for c in batch_chunks)
        padded_ids = []
        attention_masks = []
        
        for chunk in batch_chunks:
            padding_length = max_len - len(chunk)
            padded_chunk = chunk + [tokenizer.pad_token_id] * padding_length
            padded_ids.append(padded_chunk)
            
            # Attention mask: 1 for real tokens, 0 for padding
            attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 
                            for token_id in padded_chunk]
            attention_masks.append(attention_mask)
        
        # Convert to tensors
        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
        
        # BigBird-Pegasus uses block_sparse attention automatically
        # No need for global_attention_mask (that's for Longformer/LED)
        
        # Generate summary for this batch
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=gen_max,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode batch outputs
        for i in range(len(batch_windows)):
            summary = tokenizer.decode(outputs[i], skip_special_tokens=True)
            summaries.append(summary)
    
    # Aggregation strategy
    if aggregation == "hierarchical" and len(summaries) > 1:
        # Hierarchical: summarize the concatenated summaries
        combined_text = " ".join(summaries)
        
        # Tokenize the combined summaries
        combined_ids = tokenizer.encode(combined_text, add_special_tokens=False)
        
        # Truncate to BigBird capacity
        combined_ids = combined_ids[:cap]
        
        # Ensure block size alignment
        block_size = 64
        if len(combined_ids) % block_size != 0:
            padding_length = block_size - (len(combined_ids) % block_size)
            combined_ids = combined_ids + [tokenizer.pad_token_id] * padding_length
        
        # Create input tensors
        input_ids = torch.tensor([combined_ids], dtype=torch.long, device=device)
        attention_mask = torch.tensor([[1 if tid != tokenizer.pad_token_id else 0 for tid in combined_ids]], 
                                       dtype=torch.long, device=device)
        
        # Generate final summary
        with torch.no_grad():
            final_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=gen_max,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        return tokenizer.decode(final_ids[0], skip_special_tokens=True).strip()
    
    # Default: concat (original behavior)
    final_summary = " ".join(summaries)
    return final_summary
