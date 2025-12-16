"""
Test alternative fine-tuned LongT5 models for scientific summarization
"""

import sys
from src.utils.io import sample_records
from src.utils.window import make_windows, build_tokenizer
from src.models.sliding import load_model
from src.eval.metric import text_metrics
from src.utils.prof import start_prof, stop_prof

# Alternative fine-tuned LongT5 models to try
candidate_models = [
    # Scientific/academic focused
    "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps",  # PubMed (medical/scientific)
    "tau/long-t5-tglobal-base-sci-simplify",  # Scientific text simplification
    "pszemraj/long-t5-tglobal-base-sci-simplify",  # Another sci simplify
    
    # General summarization (might work)
    "google/long-t5-tglobal-base",  # Official Google (might be fine-tuned)
]

# Load 1 sample from real data
data_path = "src/data/cleaned/cleaned_extra_long_examples_arxiv.jsonl"
data = sample_records(data_path, n=1, seed=42)
doc = data[0]["text"]
ref = data[0]["ref"]

print("="*70)
print("TESTING ALTERNATIVE LONGT5 MODELS")
print("="*70)
print(f"Document: {len(doc):,} chars")
print(f"Reference: {len(ref):,} chars\n")

results = []

for model_name in candidate_models:
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print('='*70)
    
    try:
        # Load model
        print("Loading model...")
        model, tok, metadata = load_model(model_name, attention_impl="default")
        print(f"✓ Model loaded ({metadata['model_family']})")
        
        # Create windows
        base_tok = build_tokenizer(model_name)
        windows = make_windows(doc, base_tok, window_size=1024, overlap=0)
        print(f"✓ Created {len(windows)} windows")
        
        # Generate
        print("Generating summary...")
        t0 = start_prof()
        
        from src.models.sliding import generate_with_windows
        summary = generate_with_windows(model, tok, windows, gen_max=256, global_tokens=0)
        
        lat, mem = stop_prof(t0)
        
        # Evaluate
        metrics = text_metrics([summary], [ref])
        
        # Results
        print(f"\n✓ Results:")
        print(f"  ROUGE-L: {metrics['rougeL']:.4f}")
        print(f"  BERTScore: {metrics['bertscore_f1']:.4f}")
        print(f"  Length: {len(summary)} chars")
        print(f"  Latency: {lat:.2f}s")
        
        # Quality check
        is_good = (
            metrics['rougeL'] > 0.10 and
            metrics['bertscore_f1'] > 0.70 and
            len(summary) > 100
        )
        
        if is_good:
            print(f"\n  🎯 CANDIDATE FOUND!")
            print(f"  Summary preview: {summary[:200]}...")
        else:
            print(f"\n  ❌ Not good enough")
            if metrics['rougeL'] <= 0.10:
                print(f"     - ROUGE-L too low: {metrics['rougeL']:.4f}")
            if len(summary) <= 100:
                print(f"     - Too short: {len(summary)} chars")
        
        results.append({
            'model': model_name,
            'rougeL': metrics['rougeL'],
            'bertscore': metrics['bertscore_f1'],
            'length': len(summary),
            'latency': lat,
            'works': is_good
        })
        
    except Exception as e:
        print(f"✗ Error: {str(e)[:100]}")
        results.append({
            'model': model_name,
            'error': str(e)[:100],
            'works': False
        })
        continue

# Summary
print(f"\n\n{'='*70}")
print("FINAL RESULTS")
print('='*70)

working_models = [r for r in results if r.get('works', False)]

if working_models:
    print(f"\n✅ FOUND {len(working_models)} WORKING MODEL(S):\n")
    for r in working_models:
        print(f"  {r['model']}")
        print(f"    ROUGE-L: {r['rougeL']:.4f}")
        print(f"    BERTScore: {r['bertscore']:.4f}")
        print(f"    Length: {r['length']} chars")
        print(f"    Latency: {r['latency']:.2f}s")
        print()
    
    # Best model
    best = max(working_models, key=lambda x: x['rougeL'])
    print(f"🏆 BEST MODEL: {best['model']}")
    print(f"   ROUGE-L: {best['rougeL']:.4f} (LED baseline: 0.163)")
    
else:
    print("\n❌ NO WORKING MODELS FOUND")
    print("\nAll tested models failed quality checks.")
    print("\n💡 RECOMMENDATION: Use LED (allenai/led-base-16384) only")
    print("   LED ROUGE-L: 0.163")
    print("   LED BERTScore: 0.814")
    print("   LED works reliably")

print('='*70)
