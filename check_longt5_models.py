"""
Check available LongT5 models and their status
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

print("="*60)
print("Checking LongT5 Model Variants")
print("="*60)

models_to_test = [
    "google/long-t5-local-base",
    "google/long-t5-tglobal-base",
    "pszemraj/long-t5-tglobal-base-16384-book-summary",  # Fine-tuned for summarization
]

for model_name in models_to_test:
    print(f"\n--- Testing: {model_name} ---")
    try:
        # Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tok = AutoTokenizer.from_pretrained(model_name)
        
        # Check for untrained weights warning
        print(f"✓ Model loaded successfully")
        
        # Test generation
        test_text = "Summarize: Deep learning has revolutionized AI."
        inputs = tok(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        
        summary = tok.decode(outputs[0], skip_special_tokens=True)
        print(f"  Generated: {summary[:100]}")
        
        # Check if coherent
        is_gibberish = any(word in summary.lower() for word in ['einwohner', 'snack', 'difuz', 'griev'])
        print(f"  Coherent: {not is_gibberish}")
        
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print("Use a fine-tuned model for summarization, not the base model!")
