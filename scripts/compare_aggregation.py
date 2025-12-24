"""
Compare concat (from existing grid results) vs hierarchical aggregation results.
Usage: uv run python scripts/compare_aggregation.py
"""
import pandas as pd
import os
from pathlib import Path

# Mapping: hierarchical config -> corresponding concat result from grid
COMPARISON_PAIRS = [
    {
        "name": "LED_arxiv_extra_long",
        "hierarchical": "results/aggregation_test/led_arxiv_extra_long_hierarchical.csv",
        "concat": "results/grid/arxiv_extra_long/led_w1024_ov256_g16.csv"
    },
    {
        "name": "LED_longform_long", 
        "hierarchical": "results/aggregation_test/led_longform_long_hierarchical.csv",
        "concat": "results/grid/longform_long/led_w1024_ov256_g16.csv"
    },
    {
        "name": "BigBird_arxiv_extra_long",
        "hierarchical": "results/aggregation_test/bigbird_arxiv_extra_long_hierarchical.csv",
        "concat": "results/grid/arxiv_extra_long/led_w1024_ov256_g16_bigbird.csv"
    },
    {
        "name": "BigBird_longform_long",
        "hierarchical": "results/aggregation_test/bigbird_longform_long_hierarchical.csv",
        "concat": "results/grid/longform_long/led_w1024_ov256_g16_bigbird.csv"
    },
    {
        "name": "LED_arxiv_long_w2048",
        "hierarchical": "results/aggregation_test/led_arxiv_long_w2048_hierarchical.csv",
        "concat": "results/grid/arxiv_long/led_w2048_ov0_g0.csv"
    },
    {
        "name": "LED_longform_extra_long",
        "hierarchical": "results/aggregation_test/led_longform_extra_long_hierarchical.csv",
        "concat": "results/grid/longform_extra_long/led_w1024_ov256_g16.csv"
    },
]

def load_results(path):
    """Load CSV and return mean metrics."""
    if not Path(path).exists():
        return None
    try:
        df = pd.read_csv(path)
        return {
            "rougeL": df["rougeL"].mean(),
            "rougeL_std": df["rougeL"].std(),
            "bertscore_f1": df["bertscore_f1"].mean(),
            "bertscore_std": df["bertscore_f1"].std(),
            "latency": df["latency"].mean(),
            "n": len(df)
        }
    except Exception as e:
        print(f"⚠️ Error loading {path}: {e}")
        return None

def main():
    print("=" * 70)
    print("📊 AGGREGATION COMPARISON: concat vs hierarchical")
    print("=" * 70)
    print("Comparing existing concat results (grid/) with new hierarchical results")
    print("")
    
    all_concat_rouge = []
    all_hier_rouge = []
    all_concat_bert = []
    all_hier_bert = []
    
    results_found = False
    
    for pair in COMPARISON_PAIRS:
        concat = load_results(pair["concat"])
        hier = load_results(pair["hierarchical"])
        
        print(f"\n🧪 {pair['name']}")
        print("-" * 50)
        
        if concat is None:
            print(f"   ⚠️ Concat results not found: {pair['concat']}")
        else:
            print(f"   Concat (n={concat['n']}): ROUGE-L={concat['rougeL']:.4f}±{concat['rougeL_std']:.4f}, BERTScore={concat['bertscore_f1']:.4f}")
            all_concat_rouge.append(concat['rougeL'])
            all_concat_bert.append(concat['bertscore_f1'])
        
        if hier is None:
            print(f"   ⚠️ Hierarchical results not found: {pair['hierarchical']}")
            print(f"      Run: uv run python main.py --config configs/sliding/aggregation_test/{pair['name'].lower().replace('_', '_')}_hierarchical.yml")
        else:
            print(f"   Hierarchical (n={hier['n']}): ROUGE-L={hier['rougeL']:.4f}±{hier['rougeL_std']:.4f}, BERTScore={hier['bertscore_f1']:.4f}")
            all_hier_rouge.append(hier['rougeL'])
            all_hier_bert.append(hier['bertscore_f1'])
            results_found = True
        
        if concat and hier:
            rouge_diff = hier['rougeL'] - concat['rougeL']
            rouge_pct = (rouge_diff / concat['rougeL'] * 100) if concat['rougeL'] > 0 else 0
            bert_diff = hier['bertscore_f1'] - concat['bertscore_f1']
            bert_pct = (bert_diff / concat['bertscore_f1'] * 100) if concat['bertscore_f1'] > 0 else 0
            lat_diff = hier['latency'] - concat['latency']
            
            emoji_rouge = "✅" if rouge_diff > 0 else "❌"
            emoji_bert = "✅" if bert_diff > 0 else "❌"
            
            print(f"   {emoji_rouge} ROUGE-L diff: {rouge_diff:+.4f} ({rouge_pct:+.1f}%)")
            print(f"   {emoji_bert} BERTScore diff: {bert_diff:+.4f} ({bert_pct:+.1f}%)")
            print(f"   ⏱️ Latency diff: {lat_diff:+.2f}s")
    
    # Overall summary
    if all_concat_rouge and all_hier_rouge:
        print("\n" + "=" * 70)
        print("📋 OVERALL SUMMARY")
        print("=" * 70)
        
        avg_concat_rouge = sum(all_concat_rouge) / len(all_concat_rouge)
        avg_hier_rouge = sum(all_hier_rouge) / len(all_hier_rouge)
        avg_concat_bert = sum(all_concat_bert) / len(all_concat_bert)
        avg_hier_bert = sum(all_hier_bert) / len(all_hier_bert)
        
        print(f"\nAverage across {len(all_hier_rouge)} experiments:")
        print(f"  Concat:      ROUGE-L={avg_concat_rouge:.4f}, BERTScore={avg_concat_bert:.4f}")
        print(f"  Hierarchical: ROUGE-L={avg_hier_rouge:.4f}, BERTScore={avg_hier_bert:.4f}")
        
        rouge_improvement = avg_hier_rouge - avg_concat_rouge
        rouge_pct = (rouge_improvement / avg_concat_rouge * 100) if avg_concat_rouge > 0 else 0
        
        print(f"\n{'✅' if rouge_improvement > 0 else '❌'} Overall ROUGE-L change: {rouge_improvement:+.4f} ({rouge_pct:+.1f}%)")
        
        if rouge_improvement > 0:
            print("\n💡 CONCLUSION: Hierarchical aggregation improves ROUGE-L scores!")
            print("   This suggests concatenation was causing redundancy/inconsistency issues.")
        else:
            print("\n💡 CONCLUSION: Hierarchical aggregation did not improve ROUGE-L.")
            print("   The low ROUGE may be due to other factors (model capacity, task difficulty).")
    
    elif not results_found:
        print("\n" + "=" * 70)
        print("❌ No hierarchical results found yet.")
        print("Run: bash scripts/run_aggregation_test.sh")
        print("=" * 70)

if __name__ == "__main__":
    main()
