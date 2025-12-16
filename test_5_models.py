"""
Comprehensive test of 5 alternative models for long document summarization
"""

from src.utils.io import sample_records
from src.utils.window import make_windows, build_tokenizer
from src.eval.metric import text_metrics
from src.utils.prof import start_prof, stop_prof
import torch

# 5 models to test
models_to_test = [
    # LongT5 variants
    {
        "name": "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps",
        "type": "longt5",
        "description": "LongT5 fine-tuned on PubMed (medical/scientific)"
    },
    {
        "name": "tau/long-t5-tglobal-base-sci-simplify",
        "type": "longt5",
        "description": "LongT5 for scientific text simplification"
    },
    {
        "name": "pszemraj/long-t5-tglobal-base-sci-simplify",
        "type": "longt5",
        "description": "LongT5 sci-simplify (alternative)"
    },
    {
        "name": "google/long-t5-tglobal-base",
        "type": "longt5",
        "description": "Official Google LongT5 with TGlobal attention"
    },
    # BigBird
    {
        "name": "google/bigbird-pegasus-large-arxiv",
        "type": "bigbird",
        "description": "BigBird-Pegasus fine-tuned on ArXiv papers"
    }
]

# Load test data
data_path = "src/data/cleaned/cleaned_extra_long_examples_arxiv.jsonl"
data = sample_records(data_path, n=1, seed=42)
doc = data[0]["text"]
ref = data[0]["ref"]

print("="*70)
print("COMPREHENSIVE MODEL TEST - 5 Alternatives to LED")
print("="*70)
print(f"Test document: {len(doc):,} chars")
print(f"Reference: {len(ref):,} chars")
print(f"\nLED Baseline: ROUGE-L=0.163, BERTScore=0.814")
print("="*70)

results = []

for i, model_info in enumerate(models_to_test, 1):
    model_name = model_info["name"]
    model_type = model_info["type"]
    
    print(f"\n{'='*70}")
    print(f"[{i}/5] Testing: {model_name}")
    print(f"Description: {model_info['description']}")
    print('='*70)
    
    try:
        # Load model based on type
        if model_type == "longt5":
            from transformers import LongT5ForConditionalGeneration, AutoTokenizer
            print("Loading LongT5 model...")
            model = LongT5ForConditionalGeneration.from_pretrained(model_name)
            tok = AutoTokenizer.from_pretrained(model_name)
            
        elif model_type == "bigbird":
            from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
            print("Loading BigBird-Pegasus model...")
            model = BigBirdPegasusForConditionalGeneration.from_pretrained(model_name)
            tok = AutoTokenizer.from_pretrained(model_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"✓ Model loaded on {device}")
        
        # Create windows
        base_tok = build_tokenizer(model_name)
        windows = make_windows(doc, base_tok, window_size=1024, overlap=0)
        print(f"✓ Created {len(windows)} windows")
        
        # Generate summaries
        print("Generating summaries...")
        t0 = start_prof()
        
        window_summaries = []
        for j, window_ids in enumerate(windows):
            window_text = tok.decode(window_ids, skip_special_tokens=True)
            inputs = tok(window_text, return_tensors="pt", max_length=4096, truncation=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4, early_stopping=True)
            
            window_summary = tok.decode(outputs[0], skip_special_tokens=True)
            window_summaries.append(window_summary)
            
            if (j + 1) % 10 == 0:
                print(f"  Processed {j+1}/{len(windows)} windows...")
        
        summary = " ".join(window_summaries)
        lat, mem = stop_prof(t0)
        
        # Evaluate
        metrics = text_metrics([summary], [ref])
        
        # Results
        print(f"\n✓ RESULTS:")
        print(f"  ROUGE-L:    {metrics['rougeL']:.4f} (LED: 0.163)")
        print(f"  BERTScore:  {metrics['bertscore_f1']:.4f} (LED: 0.814)")
        print(f"  Length:     {len(summary)} chars")
        print(f"  Latency:    {lat:.2f}s")
        print(f"  GPU Memory: {mem:.2f}GB")
        
        # Quality check
        is_good = (
            metrics['rougeL'] > 0.10 and
            metrics['bertscore_f1'] > 0.70 and
            len(summary) > 100
        )
        
        if is_good:
            print(f"\n  ✅ PASSES quality checks")
        else:
            print(f"\n  ❌ FAILS quality checks")
            if metrics['rougeL'] <= 0.10:
                print(f"     - ROUGE-L too low: {metrics['rougeL']:.4f}")
            if metrics['bertscore_f1'] <= 0.70:
                print(f"     - BERTScore too low: {metrics['bertscore_f1']:.4f}")
            if len(summary) <= 100:
                print(f"     - Too short: {len(summary)} chars")
        
        results.append({
            'model': model_name,
            'type': model_type,
            'rougeL': metrics['rougeL'],
            'bertscore': metrics['bertscore_f1'],
            'length': len(summary),
            'latency': lat,
            'memory': mem,
            'works': is_good,
            'summary_preview': summary[:200]
        })
        
    except Exception as e:
        print(f"✗ ERROR: {str(e)[:150]}")
        results.append({
            'model': model_name,
            'type': model_type,
            'error': str(e)[:150],
            'works': False
        })

# Final summary
print(f"\n\n{'='*70}")
print("FINAL RESULTS - MODEL COMPARISON")
print('='*70)

working = [r for r in results if r.get('works', False)]

if working:
    print(f"\n✅ FOUND {len(working)} WORKING MODEL(S):\n")
    for r in sorted(working, key=lambda x: x['rougeL'], reverse=True):
        print(f"  {r['model']}")
        print(f"    ROUGE-L:    {r['rougeL']:.4f} (vs LED: 0.163, diff: {r['rougeL']-0.163:+.4f})")
        print(f"    BERTScore:  {r['bertscore']:.4f} (vs LED: 0.814, diff: {r['bertscore']-0.814:+.4f})")
        print(f"    Latency:    {r['latency']:.2f}s")
        print(f"    Preview:    {r['summary_preview']}...")
        print()
    
    best = max(working, key=lambda x: x['rougeL'])
    print(f"🏆 BEST MODEL: {best['model']}")
    print(f"   ROUGE-L: {best['rougeL']:.4f}")
    print(f"   Can be used as LED alternative!")
    
else:
    print("\n❌ NO WORKING MODELS FOUND")
    print("\nAll 5 models failed quality checks.")
    print("\n💡 FINAL RECOMMENDATION:")
    print("   Use LED (allenai/led-base-16384) ONLY")
    print("   LED ROUGE-L: 0.163")
    print("   LED BERTScore: 0.814")
    print("   LED is the most reliable baseline")

print('='*70)
