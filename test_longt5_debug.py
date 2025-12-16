"""
LongT5 Diagnostic Tests
Tests tokenizer, model loading, and generation to identify the corruption issue
"""

print("="*60)
print("TEST 1: Verify Tokenizer")
print("="*60)

from transformers import AutoTokenizer

# Load tokenizer
tok = AutoTokenizer.from_pretrained("google/long-t5-local-base")

# Test encoding/decoding
text = "This is a test sentence about deep learning and neural networks."
ids = tok.encode(text)
decoded = tok.decode(ids, skip_special_tokens=True)

print(f"Tokenizer type: {type(tok)}")
print(f"Original:  {text}")
print(f"Token IDs: {ids[:10]}... (first 10)")
print(f"Decoded:   {decoded}")
print(f"Match: {text.strip() == decoded.strip()}")

print("\n" + "="*60)
print("TEST 2: Check Model Loading")
print("="*60)

from src.models.sliding import load_model

model, tok_loaded, metadata = load_model("google/long-t5-local-base", "default")

print(f"Model type: {type(model)}")
print(f"Model class: {model.__class__.__name__}")
print(f"Tokenizer type: {type(tok_loaded)}")
print(f"Tokenizer class: {tok_loaded.__class__.__name__}")
print(f"Metadata: {metadata}")

# Check if tokenizer matches
print(f"\nTokenizer from AutoTokenizer: {type(tok)}")
print(f"Tokenizer from load_model: {type(tok_loaded)}")
print(f"Same class: {type(tok) == type(tok_loaded)}")

print("\n" + "="*60)
print("TEST 3: Test Generation Directly")
print("="*60)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load fresh model and tokenizer
model_fresh = AutoModelForSeq2SeqLM.from_pretrained("google/long-t5-local-base")
tok_fresh = AutoTokenizer.from_pretrained("google/long-t5-local-base")

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model_fresh = model_fresh.to(device)

# Test text
test_text = """
Deep learning has revolutionized artificial intelligence in recent years.
Neural networks with multiple layers can learn complex patterns from data.
Transformers have become the dominant architecture for natural language processing.
"""

# Tokenize
inputs = tok_fresh(test_text, return_tensors="pt", max_length=512, truncation=True).to(device)

print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Input tokens (first 20): {inputs['input_ids'][0][:20].tolist()}")

# Generate
with torch.no_grad():
    outputs = model_fresh.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=4,
        early_stopping=True
    )

# Decode
summary = tok_fresh.decode(outputs[0], skip_special_tokens=True)

print(f"\nGenerated summary:")
print(f"  Length: {len(summary)} chars")
print(f"  Text: {summary}")

# Check if it's coherent
is_coherent = len(summary) > 50 and not any(char in summary for char in ['Ã', 'Â', 'Ã©', 'Ã¨'])
print(f"\nCoherent: {is_coherent}")

print("\n" + "="*60)
print("TEST 4: Compare with sliding.py load_model")
print("="*60)

# Test with loaded model
inputs_loaded = tok_loaded(test_text, return_tensors="pt", max_length=512, truncation=True).to(device)

with torch.no_grad():
    outputs_loaded = model.generate(
        **inputs_loaded,
        max_new_tokens=100,
        num_beams=4,
        early_stopping=True
    )

summary_loaded = tok_loaded.decode(outputs_loaded[0], skip_special_tokens=True)

print(f"Summary from load_model:")
print(f"  Length: {len(summary_loaded)} chars")
print(f"  Text: {summary_loaded}")

print(f"\nSummaries match: {summary == summary_loaded}")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
