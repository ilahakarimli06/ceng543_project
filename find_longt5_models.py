"""
Search and test available fine-tuned LongT5 models for summarization
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Potential fine-tuned LongT5 models from HuggingFace
candidate_models = [
    # Official Google models
    "google/long-t5-tglobal-base",
    "google/long-t5-local-base",
    
    # Community fine-tuned models
    "pszemraj/long-t5-tglobal-base-16384-book-summary",
    "pszemraj/long-t5-tglobal-base-sci-simplify",
    "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps",
    "tau/long-t5-tglobal-base-sci-simplify",
]

print("="*70)
print("SEARCHING FOR FINE-TUNED LONGT5 MODELS")
print("="*70)

for model_name in candidate_models:
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print('='*70)
    
    try:
        print("Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tok = AutoTokenizer.from_pretrained(model_name)
        
        # Check for untrained weights warning
        print("✓ Model loaded")
        
        # Test generation
        test_text = """
        Deep learning has revolutionized artificial intelligence in recent years.
        Neural networks with multiple layers can learn complex patterns from data.
        Transformers have become the dominant architecture for natural language processing.
        Attention mechanisms allow models to focus on relevant parts of the input.
        """
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        inputs = tok(test_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        print("Generating summary...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=4,
                early_stopping=True
            )
        
        summary = tok.decode(outputs[0], skip_special_tokens=True)
        
        # Check quality
        is_coherent = len(summary) > 30 and not any(
            word in summary.lower() 
            for word in ['einwohner', 'snack', 'difuz', 'griev', 'nic']
        )
        
        print(f"\nGenerated Summary ({len(summary)} chars):")
        print(f"{summary}")
        print(f"\nCoherent: {'✅ YES' if is_coherent else '❌ NO'}")
        
        if is_coherent:
            print(f"\n🎯 CANDIDATE FOUND: {model_name}")
            print("This model appears to work!")
        
    except Exception as e:
        print(f"✗ Error: {str(e)[:100]}")
        continue

print("\n" + "="*70)
print("SEARCH COMPLETE")
print("="*70)
