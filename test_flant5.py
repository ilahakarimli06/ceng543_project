"""
Test script for FLAN-T5 XL Summarizer
Fine-tuned model: jordiclive/flan-t5-3b-summarizer
Uses same cleaned ArXiv data as LED experiments
"""
import yaml
import torch
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.utils.io import sample_records
from src.utils.window import make_windows
from src.utils.prof import start_prof, stop_prof
from src.eval.metric import text_metrics
import pandas as pd
import os

def load_flan_t5(model_name="jordiclive/flan-t5-3b-summarizer", device="cuda"):
    """Load FLAN-T5 fine-tuned summarizer"""
    print(f"Loading FLAN-T5 model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
    
    metadata = {
        "model_family": "FLAN-T5",
        "model_commit_hash": getattr(model.config, "_commit_hash", "unknown"),
        "tokenizer_version": __import__('transformers').__version__,
        "max_length": model.config.max_length if hasattr(model.config, 'max_length') else 512
    }
    
    return model.to(device), tokenizer, metadata

def generate_with_windows_flant5(model, tokenizer, windows, gen_max=256, device="cuda"):
    """
    Generate summaries for each window and concatenate (same as LED approach)
    FLAN-T5 max input: 512 tokens (much shorter than LED!)
    """
    cap = 512  # FLAN-T5 max input length
    summaries = []
    
    for window_ids in windows:
        # Truncate to max length
        chunk = window_ids[:cap]
        
        # Convert to tensor
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate summary for this window
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=gen_max,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary)
    
    # Concatenate all window summaries
    final_summary = " ".join(summaries)
    return final_summary

def run_flant5_test(config_path):
    """Run FLAN-T5 test with same setup as LED"""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Override model
    model_name = "jordiclive/flan-t5-3b-summarizer"
    
    # Setup paths
    out_path = cfg.get("out_csv", "results/output.csv").replace("led_", "flant5_")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    rows = []
    samples_data = []
    
    # Load data (same as LED)
    data = sample_records(cfg["dataset_path"], cfg["samples"], cfg.get("seed", 42))
    
    # Load model once
    model, tok, metadata = load_flan_t5(model_name)
    
    for idx, ex in enumerate(data):
        doc = ex.get("text", "")
        ref = ex.get("ref", "")
        
        t0 = start_prof()
        
        # Create windows using FLAN-T5 tokenizer
        base_tok = AutoTokenizer.from_pretrained(model_name)
        wins = make_windows(doc, base_tok, cfg["window_size"], cfg["overlap"])
        
        # Generate summary
        pred = generate_with_windows_flant5(model, tok, wins, cfg["gen_max_tokens"])
        
        lat, mem = stop_prof(t0)
        mets = text_metrics([pred], [ref])
        throughput = 3600 / lat if lat > 0 else 0
        
        # Store for CSV
        rows.append({
            "category": cfg.get("category", "unknown"),
            "domain": cfg.get("domain", "unknown"),
            "length_category": cfg.get("length_category", "unknown"),
            "model_family": "FLAN-T5",
            "model": model_name,
            "window": cfg["window_size"],
            "overlap": cfg["overlap"],
            "latency": lat,
            "gpu_peak_gb": mem,
            "throughput_docs_per_hour": throughput,
            "seed": cfg.get("seed", 42),
            "n_samples": cfg["samples"],
            **mets
        })
        
        # Store for JSON
        samples_data.append({
            "sample_id": idx,
            "doc_id": ex.get("id", f"doc_{idx}"),
            "source_preview": doc[:500],
            "reference": ref,
            "generated": pred,
            "metrics": {
                "rougeL": mets["rougeL"],
                "bertscore_f1": mets["bertscore_f1"],
                "latency": lat
            }
        })
        
        # Incremental save every 10 docs
        if (idx + 1) % 10 == 0 or (idx + 1) == len(data):
            df_temp = pd.DataFrame(rows)
            df_temp.to_csv(out_path, index=False)
            print(f"💾 Progress saved: {idx + 1}/{len(data)} documents")
    
    # Final save
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"✅ Results saved: {out_path}")
    
    # Save JSON samples
    samples_path = out_path.replace(".csv", "_samples.json")
    sorted_by_rouge = sorted(samples_data, key=lambda x: x["metrics"]["rougeL"])
    sorted_by_bert = sorted(samples_data, key=lambda x: x["metrics"]["bertscore_f1"])
    
    selected_samples = []
    selected_ids = set()
    
    # First 10 + worst 5 ROUGE + worst 5 BERTScore
    for sample in samples_data[:10]:
        selected_samples.append({**sample, "selection_reason": "first_10"})
        selected_ids.add(sample["sample_id"])
    
    for sample in sorted_by_rouge[:5]:
        if sample["sample_id"] not in selected_ids:
            selected_samples.append({**sample, "selection_reason": "worst_rouge"})
            selected_ids.add(sample["sample_id"])
    
    for sample in sorted_by_bert[:5]:
        if sample["sample_id"] not in selected_ids:
            selected_samples.append({**sample, "selection_reason": "worst_bertscore"})
            selected_ids.add(sample["sample_id"])
    
    json_output = {
        "config": {
            "model": model_name,
            "model_family": "FLAN-T5",
            "window_size": cfg["window_size"],
            "overlap": cfg["overlap"],
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
    import argparse
    parser = argparse.ArgumentParser(description="Test FLAN-T5 summarizer")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    run_flant5_test(args.config)
