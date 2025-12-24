import yaml, torch
import argparse
import json
from src.utils.io import sample_records
from src.utils.window import make_windows, build_tokenizer
from src.utils.prof import start_prof, stop_prof
from src.eval.metric import text_metrics
from src.models.sliding import load_model, generate_with_windows

def run_sliding(cfg, attention_override=None, model_override=None):
    # Override attention_impl if specified via command line
    if attention_override:
        cfg["attention_impl"] = attention_override
    
    # Override model_name if specified via command line
    if model_override:
        cfg["model_name"] = model_override
    
    rows = []
    samples_data = []  # Store summaries for JSON output
    
    # Setup paths
    import pandas as pd
    out_path = cfg.get("out_csv", "results/output.csv")
    if attention_override and attention_override != "default":
        out_path = out_path.replace(".csv", f"_{attention_override}.csv")
    if model_override:
        # Detect model family from model name
        if "bigbird" in model_override.lower():
            model_suffix = "bigbird"
        else:
            model_suffix = "led"
        out_path = out_path.replace(".csv", f"_{model_suffix}.csv")
    
    # Add aggregation suffix to output path if hierarchical
    aggregation = cfg.get("aggregation", "concat")
    if aggregation == "hierarchical":
        out_path = out_path.replace(".csv", "_hierarchical.csv")
    
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Determine attention implementation
    attn_impl = attention_override if attention_override else cfg.get("attention_impl", "default")
    
    # Load model ONCE before processing documents (critical for performance)
    model, tok, metadata = load_model(cfg["model_name"], attn_impl)
    
    data = sample_records(cfg["dataset_path"], cfg["samples"], cfg.get("seed",42))
    for idx, ex in enumerate(data):
        doc = ex.get("text", "")
        ref = ex.get("ref", "")
        t0 = start_prof()
        base_tok = build_tokenizer(cfg["model_name"])
        wins = make_windows(doc, base_tok, cfg["window_size"], cfg["overlap"])
        pred = generate_with_windows(model, tok, wins, cfg["gen_max_tokens"], cfg.get("global_tokens",0), aggregation=aggregation)
        lat, mem = stop_prof(t0)
        mets = text_metrics([pred],[ref])
        throughput = 3600 / lat if lat > 0 else 0  # docs per hour
        
        # Store for CSV
        rows.append({
          "category": cfg.get("category", "unknown"),
          "domain": cfg.get("domain", "unknown"),
          "length_category": cfg.get("length_category", "unknown"),
          "model_family": metadata.get("model_family", "unknown"),
          "model": cfg["model_name"],
          "attention": cfg.get("attention_impl", "default"),
          "attention_backend": metadata.get("attention_backend", "unknown"),
          "flash_attn_version": metadata.get("flash_attn_version", "N/A"),
          "window": cfg["window_size"], "overlap": cfg["overlap"],
          "global_tokens": cfg.get("global_tokens", 0),
          "aggregation": aggregation,
          "latency": lat, "gpu_peak_gb": mem,
          "throughput_docs_per_hour": throughput,
          "seed": cfg.get("seed", 42),
          "n_samples": cfg["samples"],
          "model_commit_hash": metadata.get("model_commit_hash", "unknown"),
          "tokenizer_version": metadata.get("tokenizer_version", "unknown"),
          **mets
        })
        
        # Store for JSON (all samples, will filter later)
        samples_data.append({
            "sample_id": idx,
            "doc_id": ex.get("id", f"doc_{idx}"),
            "source_preview": doc[:500],  # First 500 chars
            "source_length": len(doc),
            "reference": ref,
            "generated": pred,
            "metrics": {
                "rougeL": mets["rougeL"],
                "bertscore_f1": mets["bertscore_f1"],
                "latency": lat
            }
        })
        
        # INCREMENTAL SAVE: Save every 10 documents (or at the end)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(data):
            df_temp = pd.DataFrame(rows)
            df_temp.to_csv(out_path, index=False)
            print(f"💾 Progress saved: {idx + 1}/{len(data)} documents")
    
    # Final CSV save (already done incrementally, but ensure it's complete)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"✅ Results saved: {out_path}")
    
    # Save JSON samples (smart sampling)
    samples_path = out_path.replace(".csv", "_samples.json")
    
    # Sort by ROUGE-L and BERTScore to find worst cases
    sorted_by_rouge = sorted(samples_data, key=lambda x: x["metrics"]["rougeL"])
    sorted_by_bert = sorted(samples_data, key=lambda x: x["metrics"]["bertscore_f1"])
    
    # Collect samples: first 10 + worst 5 ROUGE + worst 5 BERTScore
    selected_samples = []
    selected_ids = set()
    
    # First 10 samples
    for sample in samples_data[:10]:
        selected_samples.append({**sample, "selection_reason": "first_10"})
        selected_ids.add(sample["sample_id"])
    
    # Worst 5 by ROUGE-L
    for sample in sorted_by_rouge[:5]:
        if sample["sample_id"] not in selected_ids:
            selected_samples.append({**sample, "selection_reason": "worst_rouge"})
            selected_ids.add(sample["sample_id"])
    
    # Worst 5 by BERTScore
    for sample in sorted_by_bert[:5]:
        if sample["sample_id"] not in selected_ids:
            selected_samples.append({**sample, "selection_reason": "worst_bertscore"})
            selected_ids.add(sample["sample_id"])
    
    # Save to JSON
    json_output = {
        "config": {
            "model": cfg["model_name"],
            "model_family": metadata.get("model_family", "unknown"),
            "attention": cfg.get("attention_impl", "default"),
            "window_size": cfg["window_size"],
            "overlap": cfg["overlap"],
            "global_tokens": cfg.get("global_tokens", 0),
            "aggregation": aggregation,
            "category": cfg.get("category", "unknown")
        },
        "summary_stats": {
            "total_samples": len(samples_data),
            "avg_rougeL": df["rougeL"].mean(),
            "avg_bertscore": df["bertscore_f1"].mean(),
            "min_rougeL": df["rougeL"].min(),
            "min_bertscore": df["bertscore_f1"].min()
        },
        "samples": selected_samples
    }
    
    with open(samples_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Samples saved: {samples_path} ({len(selected_samples)} examples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sliding window experiments")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--attention", choices=["default", "flash_attention_2"], 
                       help="Override attention implementation (optional)")
    parser.add_argument("--model", 
                       help="Override model name (e.g., google/long-t5-local-base)")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    if cfg["method"]=="sliding": 
        run_sliding(cfg, attention_override=args.attention, model_override=args.model)
