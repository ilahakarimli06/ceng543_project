"""
Proper LongT5 test using REAL ArXiv data and sliding window pipeline
"""

import sys
import yaml
from src.utils.io import sample_records
from src.utils.window import make_windows, build_tokenizer
from src.models.sliding import load_model
from src.eval.metric import text_metrics
from src.utils.prof import start_prof, stop_prof

print("="*70)
print("PROPER LONGT5 TEST - Using Real ArXiv Data")
print("="*70)

# Model name (already downloaded)
model_name = "pszemraj/long-t5-tglobal-base-16384-book-summary"

# Load 1 sample from real data
data_path = "src/data/cleaned/cleaned_extra_long_examples_arxiv.jsonl"
data = sample_records(data_path, n=1, seed=42)

doc = data[0]["text"]
ref = data[0]["ref"]

print(f"\nDocument info:")
print(f"  Source length: {len(doc):,} chars")
print(f"  Reference length: {len(ref):,} chars")

# Load model using sliding.py (same as LED)
print(f"\nLoading model: {model_name}")
print("(Model already downloaded, loading from cache...)")

try:
    model, tok, metadata = load_model(model_name, attention_impl="default")
    
    print(f"✓ Model loaded")
    print(f"  Model family: {metadata['model_family']}")
    print(f"  Attention backend: {metadata['attention_backend']}")
    
    # Create windows (same as LED experiments)
    print(f"\nCreating windows...")
    base_tok = build_tokenizer(model_name)
    window_size = 1024
    overlap = 0
    
    windows = make_windows(doc, base_tok, window_size, overlap)
    print(f"  ✓ Created {len(windows)} windows")
    
    # Generate summary using sliding window approach
    print(f"\nGenerating summary...")
    t0 = start_prof()
    
    # Import generation function
    from src.models.sliding import generate_with_windows
    
    summary = generate_with_windows(
        model, tok, windows, 
        gen_max=256,  # Same as LED
        global_tokens=0,
        device="cuda"
    )
    
    lat, mem = stop_prof(t0)
    
    print(f"  ✓ Generated summary")
    print(f"  Length: {len(summary)} chars")
    print(f"  Latency: {lat:.2f}s")
    print(f"  GPU Memory: {mem:.2f}GB")
    
    # Evaluate
    print(f"\nEvaluating...")
    metrics = text_metrics([summary], [ref])
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print('='*70)
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")
    print(f"BERTScore: {metrics['bertscore_f1']:.4f}")
    print(f"Latency: {lat:.2f}s")
    print(f"GPU Memory: {mem:.2f}GB")
    
    print(f"\n{'='*70}")
    print("GENERATED SUMMARY (first 500 chars):")
    print('='*70)
    print(summary[:500])
    print("...")
    
    print(f"\n{'='*70}")
    print("REFERENCE SUMMARY (first 500 chars):")
    print('='*70)
    print(ref[:500])
    print("...")
    
    # Quality check
    print(f"\n{'='*70}")
    print("QUALITY CHECK")
    print('='*70)
    
    is_good = (
        metrics['rougeL'] > 0.10 and  # At least 10% overlap
        metrics['bertscore_f1'] > 0.70 and  # Reasonable semantic similarity
        len(summary) > 100  # Not too short
    )
    
    print(f"ROUGE-L > 0.10: {'✅' if metrics['rougeL'] > 0.10 else '❌'} ({metrics['rougeL']:.4f})")
    print(f"BERTScore > 0.70: {'✅' if metrics['bertscore_f1'] > 0.70 else '❌'} ({metrics['bertscore_f1']:.4f})")
    print(f"Length > 100 chars: {'✅' if len(summary) > 100 else '❌'} ({len(summary)} chars)")
    
    print(f"\n{'='*70}")
    if is_good:
        print("✅ MODEL WORKS! Can be used for experiments")
        print(f"\nComparison with LED baseline:")
        print(f"  LED ROUGE-L: 0.163 (average)")
        print(f"  LongT5 ROUGE-L: {metrics['rougeL']:.4f}")
        print(f"  Difference: {metrics['rougeL'] - 0.163:+.4f}")
    else:
        print("❌ MODEL DOES NOT WORK - Use LED only")
        print(f"\nIssues:")
        if metrics['rougeL'] <= 0.10:
            print(f"  - ROUGE-L too low ({metrics['rougeL']:.4f})")
        if metrics['bertscore_f1'] <= 0.70:
            print(f"  - BERTScore too low ({metrics['bertscore_f1']:.4f})")
        if len(summary) <= 100:
            print(f"  - Summary too short ({len(summary)} chars)")
    print('='*70)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n❌ Model failed to load or generate")
    print("Recommendation: Use LED only")
