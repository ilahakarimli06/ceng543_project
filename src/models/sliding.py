from transformers import LEDForConditionalGeneration, LongT5ForConditionalGeneration, AutoTokenizer
import torch

def load_model(name, attention_impl="default", device="cuda"):
    """
    Load encoder-decoder model with configurable attention implementation.
    
    Supports:
        - LED (BART-based): allenai/led-base-16384
        - LongT5 (T5-based): google/long-t5-local-base
    
    Args:
        attention_impl: "default" (standard PyTorch) or "flash_attention_2" (optimized)
    
    Returns:
        model, tokenizer, metadata dict (for reproducibility logging)
    """
    metadata = {
        "attention_impl": attention_impl,
        "attention_backend": "unknown",
        "flash_attn_version": None,
        "model_family": "unknown",
    }
    
    # Detect model type
    name_lower = name.lower()
    is_led = "led" in name_lower
    is_longt5 = "long-t5" in name_lower or "longt5" in name_lower
    
    if is_led:
        metadata["model_family"] = "LED"
        # Determine attention implementation
        model_kwargs = {"dtype": "auto"}
        
        if attention_impl == "flash_attention_2":
            try:
                # Try to use FlashAttention-2
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model = LEDForConditionalGeneration.from_pretrained(name, **model_kwargs)
                metadata["attention_backend"] = "flash_attention_2"
                
                # Log flash-attn version if available
                try:
                    import flash_attn
                    metadata["flash_attn_version"] = flash_attn.__version__
                except ImportError:
                    metadata["flash_attn_version"] = "unknown"
                    
                print(f"✓ Using FlashAttention-2 (flash-attn {metadata['flash_attn_version']})")
                
            except Exception as e:
                # Fallback to standard attention
                print(f"⚠ FlashAttention-2 not available ({e}), falling back to standard attention")
                model = LEDForConditionalGeneration.from_pretrained(name)
                metadata["attention_backend"] = "eager"
                metadata["attention_impl"] = "default"  # Update to reflect actual
        else:
            # Standard PyTorch attention
            model = LEDForConditionalGeneration.from_pretrained(name)
            metadata["attention_backend"] = "eager"
        
        tok = AutoTokenizer.from_pretrained(name)
        
    elif is_longt5:
        metadata["model_family"] = "LongT5"
        # LongT5 uses local attention only (no flash attention support yet)
        model = LongT5ForConditionalGeneration.from_pretrained(name)
        metadata["attention_backend"] = "local"  # LongT5's built-in local attention
        tok = AutoTokenizer.from_pretrained(name)
        print(f"✓ Using LongT5 with local attention")
        
    else:
        raise ValueError(f"Unsupported model: {name}. Use LED or LongT5 models.")

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
    
    return model.to(device), tok, metadata

def generate_with_windows(model, tokenizer, windows, gen_max=256, global_tokens=0, device="cuda"):
    """
    Her pencere (window) için ayrı ayrı özet üretir ve bunları birleştirir.
    Token-ID seviyesinde çalışır - decode/re-tokenize akışını kullanmaz.
    
    Args:
        global_tokens: LED için ilk N token'a global attention ver (0, 16, 64)
    """
    model.eval()
    outputs = []

    # LED kapasitesi belirleme
    led_cap = 16384 if getattr(model.config, "model_type", "") == "led" else None
    cfg_cap = getattr(model.config, "max_position_embeddings", None)
    tok_cap = getattr(tokenizer, "model_max_length", None)
    cap = next((c for c in [led_cap, cfg_cap, tok_cap, 4096] if isinstance(c, int) and c > 0), 4096)

    # LED encoder için special token ekleme
    is_led = getattr(model.config, "model_type", "") == "led"

    for window_ids in windows:
        # Token ID'leri üzerinde doğrudan çalış
        # LED encoder input'u special token gerektirmez, sadece dilimle
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
        
        # Tensor'a çevir ve batch dimension ekle
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        
        # Attention mask oluştur (tüm token'lar valid)
        attention_mask = torch.ones_like(input_ids)
        
        # Input dict oluştur
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # LED ise global attention mask ayarla (SMART PLACEMENT)
        # LongT5 uses local attention only - no global attention mask needed
        is_led_model = hasattr(model.config, "model_type") and "led" in model.config.model_type.lower()
        
        if is_led_model and global_tokens > 0:
            global_attention_mask = torch.zeros_like(input_ids)
            
            seq_len = input_ids.size(1)
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
                if pos < seq_len:
                    global_attention_mask[0, pos] = 1
            
            inputs["global_attention_mask"] = global_attention_mask

        # Generate
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=gen_max,
                num_beams=4,
                early_stopping=True
            )

        # Decode
        summary = tokenizer.decode(
            out_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        outputs.append(summary.strip())

    return " ".join(outputs)[:2048]
