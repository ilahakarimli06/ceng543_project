"""
Quick test of the most promising fine-tuned LongT5 model
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

print("="*70)
print("TESTING: pszemraj/long-t5-tglobal-base-16384-book-summary")
print("="*70)
print("This model is fine-tuned for book summarization (long documents)")
print()

model_name = "pszemraj/long-t5-tglobal-base-16384-book-summary"

try:
    print("Loading model (this may take a few minutes)...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"✓ Model loaded on {device}")
    print(f"  Model class: {model.__class__.__name__}")
    print(f"  Tokenizer class: {tok.__class__.__name__}")
    
    # Test with academic text (similar to ArXiv)
    test_text = """
    Deep learning has revolutionized artificial intelligence in recent years.
    Neural networks with multiple layers can learn complex patterns from large datasets.
    Convolutional neural networks excel at computer vision tasks such as image classification.
    Recurrent neural networks and transformers are effective for sequential data processing.
    The transformer architecture, introduced in 2017, has become dominant in natural language processing.
    Attention mechanisms allow models to focus on relevant parts of the input sequence.
    Large language models like GPT and BERT demonstrate impressive capabilities across various tasks.
    However, these models require substantial computational resources for training and inference.
    Research continues on making AI more efficient, interpretable, and accessible.
    """
    
    print("\nGenerating summary...")
    inputs = tok(test_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    summary = tok.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n{'='*70}")
    print("GENERATED SUMMARY:")
    print('='*70)
    print(summary)
    print('='*70)
    
    # Quality check
    is_coherent = (
        len(summary) > 30 and 
        not any(word in summary.lower() for word in ['einwohner', 'snack', 'difuz', 'griev']) and
        any(word in summary.lower() for word in ['learning', 'neural', 'model', 'network', 'ai', 'deep'])
    )
    
    print(f"\nQuality Check:")
    print(f"  Length: {len(summary)} chars")
    print(f"  Coherent: {'✅ YES' if is_coherent else '❌ NO'}")
    print(f"  Contains relevant keywords: {'✅ YES' if is_coherent else '❌ NO'}")
    
    if is_coherent:
        print(f"\n🎯 SUCCESS! This model works for summarization!")
        print(f"\nRecommendation:")
        print(f"  Use this model instead of google/long-t5-local-base")
        print(f"  Model: {model_name}")
    else:
        print(f"\n❌ This model also has issues")
        
except Exception as e:
    print(f"\n✗ Error loading model: {e}")
    print("\nTrying alternative model...")
    
    # Fallback to tglobal-base (official Google model)
    print("\n" + "="*70)
    print("TESTING: google/long-t5-tglobal-base")
    print("="*70)
    
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained("google/long-t5-tglobal-base")
        tok = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        print(f"✓ Model loaded on {device}")
        
        # Same test
        inputs = tok(test_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, num_beams=4)
        
        summary = tok.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nGenerated: {summary}")
        
    except Exception as e2:
        print(f"✗ Also failed: {e2}")
        print("\n💡 Recommendation: Stick with LED (allenai/led-base-16384)")
