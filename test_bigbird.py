"""
Test script for BigBird-Pegasus ArXiv
Fine-tuned model: google/bigbird-pegasus-large-arxiv
Specifically trained on ArXiv papers - perfect for our dataset!
"""
import yaml
import torch
import json
from transformers import AutoTokenizer, BigBirdPegasusForConditionalGeneration
from src.utils.io import sample_records
from src.utils.window import make_windows
from src.utils.prof import start_prof, stop_prof
from src.eval.metric import text_metrics
import pandas as pd
import os

def load_bigbird_pegasus(model_name="google/bigbird-pegasus-large-arxiv", device="cuda"):
    """Load BigBird-Pegasus ArXiv model"""
    print(f"Loading BigBird-Pegasus model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # BigBird-Pegasus specific: use block_sparse attention for long docs
    model = BigBirdPegasusForConditionalGeneration.from_pretrained(
        model_name,
        attention_type="block_sparse"  # Efficient for long sequences
    )
    
    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
    
    metadata = {
        "model_family": "BigBird-Pegasus",
        "model_commit_hash": getattr(model.config, "_commit_hash", "unknown"),
        "tokenizer_version": __import__('transformers').__version__,
        "max_length": 4096,  # BigBird-Pegasus max length
        "attention_type": "block_sparse",
        "block_size": model.config.block_size if hasattr(model.config, 'block_size') else 64
    }
    
    return model.to(device), tokenizer, metadata

def generate_with_windows_bigbird(model, tokenizer, windows, gen_max=256, device="cuda"):
    """
    Generate summaries for each window and concatenate
    BigBird-Pegasus max input: 4096 tokens
    """
    cap = 4096  # BigBird-Pegasus max input length
    summaries = []
    
    for window_ids in windows:
        # Truncate to max length
        chunk = window_ids[:cap]
        
        # Ensure length is divisible by block_size (64)
        block_size = 64
        if len(chunk) % block_size != 0:
            # Pad to nearest multiple of block_size with pad_token_id
            padding_length = block_size - (len(chunk) % block_size)
            chunk = chunk + [tokenizer.pad_token_id] * padding_length
        
        # Convert to tensor
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        
        # Proper attention mask: 1 for real tokens, 0 for padding
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        
        # BigBird-Pegasus: Add global attention mask (EVENLY SPACED)
        # More stable than boundary detection
        global_attention_mask = torch.zeros_like(input_ids)
        
        seq_len = len(chunk)
        # Evenly spaced positions: start, 1/4, 1/2, 3/4, end
        positions = [
            0,
            seq_len // 4,
            seq_len // 2,
            3 * seq_len // 4,
            seq_len - 1
        ]
        
        for pos in positions:
            if pos < seq_len:
                global_attention_mask[0, pos] = 1
        
        # Generate summary for this window
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
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

def run_bigbird_test(config_path):
    """Run BigBird-Pegasus test with same setup as LED"""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Override model
    model_name = "google/bigbird-pegasus-large-arxiv"
    
    # Setup paths
    out_path = cfg.get("out_csv", "results/output.csv").replace("led_", "bigbird_")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    rows = []
    samples_data = []
    
    # Load data (same as LED)
    data = sample_records(cfg["dataset_path"], cfg["samples"], cfg.get("seed", 42))
    
    # Load model once
    model, tok, metadata = load_bigbird_pegasus(model_name)
    
    for idx, ex in enumerate(data):
        doc = ex.get("text", "")
        ref = ex.get("ref", "")
        
        t0 = start_prof()
        
        # Create windows using BigBird tokenizer
        base_tok = AutoTokenizer.from_pretrained(model_name)
        wins = make_windows(doc, base_tok, cfg["window_size"], cfg["overlap"])
        
        # Generate summary
        pred = generate_with_windows_bigbird(model, tok, wins, cfg["gen_max_tokens"])
        
        lat, mem = stop_prof(t0)
        mets = text_metrics([pred], [ref])
        throughput = 3600 / lat if lat > 0 else 0
        
        # Store for CSV
        rows.append({
            "category": cfg.get("category", "unknown"),
            "domain": cfg.get("domain", "unknown"),
            "length_category": cfg.get("length_category", "unknown"),
            "model_family": "BigBird-Pegasus",
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
            "model_family": "BigBird-Pegasus",
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
    parser = argparse.ArgumentParser(description="Test BigBird-Pegasus ArXiv")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    run_bigbird_test(args.config)
