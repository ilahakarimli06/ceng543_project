"""
PLAID Benchmark Entry Point.

Runs benchmarks for PLAID token pruning + compression retrieval:
- Token-level embeddings with BERT (ColBERT style)
- Token importance scoring and pruning
- PQ compression for memory efficiency
- Comparison with full ColBERT

Usage:
    # Run full benchmark
    python main_plaid_benchmark.py --config configs/plaid_benchmark/full_grid.yml
    
    # Quick test with few samples
    python main_plaid_benchmark.py --config configs/plaid_benchmark/full_grid.yml --samples 5
"""

import yaml
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm

from src.utils.io import sample_records
from src.utils.chunker import chunk_text
from src.utils.window import build_tokenizer
from src.utils.prof import start_prof, stop_prof
from src.eval.metric import text_metrics
from src.models.sliding import load_model
from src.models.retriever import create_query_from_document
from src.models.retriever_advanced import compute_ground_truth_chunks
from src.models.generator import concatenate_chunks, generate_from_chunks
from src.models.colbert_retriever import (
    load_colbert_model,
    build_token_index,
    get_colbert_index_size_mb
)
from src.models.plaid_retriever import (
    build_plaid_index,
    plaid_retrieve_and_benchmark,
    get_plaid_index_size_mb
)


def run_plaid_benchmark(cfg: Dict, samples_override: Optional[int] = None):
    """
    Run PLAID retrieval benchmarks.
    
    Pipeline:
    1. Load documents and chunk them
    2. Build ColBERT token-level index
    3. Apply PLAID pruning + compression
    4. Compute ground-truth using ROUGE-L
    5. Benchmark retrieval (Recall@K, MRR, latency, index size)
    6. Generate summaries with top-K chunks
    7. Evaluate downstream ROUGE-L
    
    Args:
        cfg: Configuration dict from YAML
        samples_override: Override number of samples
    """
    print("=" * 60)
    print("🔬 PLAID Token Pruning + Compression Benchmark")
    print("=" * 60)
    
    # Extract config
    samples = samples_override or cfg.get("samples", 60)
    seed = cfg.get("seed", 42)
    dataset_path = cfg["dataset_path"]
    
    # Chunking config
    chunk_size = cfg.get("chunk_size", 1024)
    overlap = cfg.get("overlap", 0)
    segmentation_method = cfg.get("segmentation_method", "fixed")
    
    # Retrieval config
    top_k = cfg.get("top_k", 10)
    ground_truth_top_n = cfg.get("ground_truth_top_n", 5)
    query_strategy = cfg.get("query_strategy", "first_tokens")
    
    # ColBERT/PLAID config
    colbert_model_name = cfg.get("colbert_model", "bert-base-uncased")
    max_token_length = cfg.get("max_token_length", 512)
    
    # PLAID configurations to test
    plaid_configs = cfg.get("plaid_configs", [
        {"tokens_per_chunk": 32, "M": 48, "nbits": 8},
        {"tokens_per_chunk": 32, "M": 96, "nbits": 8},
        {"tokens_per_chunk": 16, "M": 48, "nbits": 8},
        {"tokens_per_chunk": 64, "M": 48, "nbits": 8},
    ])
    
    # Summarization model
    summarization_model = cfg.get("summarization_model", "google/bigbird-pegasus-large-arxiv")
    
    # Output paths
    out_dir = cfg.get("out_dir", "results/plaid_benchmark")
    os.makedirs(out_dir, exist_ok=True)
    
    cache_dir = cfg.get("cache_dir", "cache/plaid_benchmark")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"\n📁 Dataset: {dataset_path}")
    print(f"📊 Samples: {samples}")
    print(f"🔧 Chunk size: {chunk_size}, Overlap: {overlap}")
    print(f"🎯 Top-K: {top_k}, Ground-truth top-N: {ground_truth_top_n}")
    print(f"🤖 ColBERT model: {colbert_model_name}")
    print(f"📦 PLAID configs: {len(plaid_configs)} configurations")
    print(f"📝 Summarization model: {summarization_model}")
    
    # Load data
    print(f"\n📥 Loading {samples} samples...")
    np.random.seed(seed)
    data = sample_records(dataset_path, samples, seed)
    
    # Load ColBERT model
    print(f"\n🔄 Loading ColBERT model: {colbert_model_name}...")
    colbert_model, colbert_tok = load_colbert_model(colbert_model_name)
    
    # Load summarization model
    print(f"\n🤖 Loading summarization model: {summarization_model}...")
    base_tok = build_tokenizer(summarization_model)
    sum_model, sum_tok, sum_metadata = load_model(summarization_model, "default")
    print(f"  ✓ Model loaded (family: {sum_metadata.get('model_family', 'unknown')})")
    
    # Results storage
    all_results = []
    
    # Process each document
    print(f"\n🔍 Processing documents...")
    for doc_idx, ex in enumerate(tqdm(data, desc="Documents")):
        doc = ex.get("text", "")
        ref = ex.get("ref", "")
        
        if not doc.strip() or not ref.strip():
            continue
        
        # Step 1: Chunk document
        chunks = chunk_text(doc, base_tok, chunk_size, overlap, method=segmentation_method)
        
        if len(chunks) < 2:
            continue
        
        # Step 2: Build ColBERT token index (once per document)
        colbert_index = build_token_index(
            chunks, colbert_model, colbert_tok,
            max_length=max_token_length,
            batch_size=32
        )
        
        colbert_size_mb = get_colbert_index_size_mb(colbert_index)
        
        # Step 3: Compute ground-truth
        gt_indices, gt_scores = compute_ground_truth_chunks(chunks, ref, ground_truth_top_n)
        
        # Step 4: Create query
        query_text = create_query_from_document(doc, base_tok, strategy=query_strategy)
        
        # Step 5: Test each PLAID configuration
        for plaid_cfg in plaid_configs:
            tokens_per_chunk = plaid_cfg.get("tokens_per_chunk", 32)
            M = plaid_cfg.get("M", 48)
            nbits = plaid_cfg.get("nbits", 8)
            
            # Build PLAID index
            plaid_index = build_plaid_index(
                colbert_index, chunks, colbert_tok,
                tokens_per_chunk=tokens_per_chunk,
                M=M,
                nbits=nbits,
                seed=seed
            )
            
            plaid_size_mb = get_plaid_index_size_mb(plaid_index)
            
            # Retrieve and benchmark
            bench_results = plaid_retrieve_and_benchmark(
                query_text, chunks, plaid_index,
                colbert_model, colbert_tok,
                top_k, gt_indices
            )
            
            # Generate summary
            retrieved_indices = bench_results["retrieved_indices"][:top_k]
            
            t0 = start_prof()
            context_text = concatenate_chunks(
                chunks, retrieved_indices, base_tok, 
                max_tokens=16000, preserve_order=True
            )
            pred = generate_from_chunks(
                sum_model, sum_tok, context_text, 
                cfg.get("gen_max_tokens", 512)
            )
            lat, mem = stop_prof(t0)
            
            # Evaluate downstream ROUGE-L
            mets = text_metrics([pred], [ref])
            
            # Build result row
            result = {
                "doc_idx": doc_idx,
                "colbert_model": colbert_model_name,
                "tokens_per_chunk": tokens_per_chunk,
                "M": M,
                "nbits": nbits,
                "num_chunks": len(chunks),
                "num_tokens_original": colbert_index.num_tokens,
                "num_tokens_pruned": plaid_index.num_tokens,
                "compression_ratio": plaid_index.compression_ratio,
                "gt_top_n": ground_truth_top_n,
                "top_k": top_k,
                "gt_valid": bench_results.get("gt_valid", True),
                "recall_at_1": bench_results["recall_at_1"],
                "recall_at_5": bench_results["recall_at_5"],
                "recall_at_10": bench_results["recall_at_10"],
                "recall_at_20": bench_results["recall_at_20"],
                "mrr": bench_results["mrr"],
                "retrieval_latency_ms": bench_results["latency_ms"],
                "colbert_index_size_mb": colbert_size_mb,
                "plaid_index_size_mb": plaid_size_mb,
                "size_reduction_ratio": colbert_size_mb / plaid_size_mb if plaid_size_mb > 0 else 0,
                "downstream_rouge_l": mets["rougeL"],
                "downstream_bertscore": mets["bertscore_f1"],
                "total_latency_s": lat,
                "gpu_peak_gb": mem
            }
            
            all_results.append(result)
    
    # Save results
    df = pd.DataFrame(all_results)
    
    # Per-document results
    out_path = os.path.join(out_dir, "benchmark_full.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved per-document results: {out_path}")
    
    # Aggregated by configuration
    if len(df) > 0:
        agg_columns = ["colbert_model", "tokens_per_chunk", "M", "nbits"]
        metric_columns = [
            "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20",
            "mrr", "retrieval_latency_ms", "plaid_index_size_mb",
            "size_reduction_ratio", "compression_ratio",
            "downstream_rouge_l", "downstream_bertscore"
        ]
        
        agg_df = df.groupby(agg_columns)[metric_columns].mean().reset_index()
        agg_df = agg_df.round(4)
        
        agg_path = os.path.join(out_dir, "benchmark_aggregated.csv")
        agg_df.to_csv(agg_path, index=False)
        print(f"✅ Saved aggregated results: {agg_path}")
        
        # Print summary
        print("\n" + "=" * 100)
        print("📊 PLAID BENCHMARK SUMMARY (Aggregated)")
        print("=" * 100)
        print(agg_df.to_string(index=False))
        
        # Save summary JSON
        summary = {
            "config": cfg,
            "plaid_configs": plaid_configs,
            "num_documents": len(data),
            "aggregated_results": agg_df.to_dict(orient="records")
        }
        
        summary_path = os.path.join(out_dir, "benchmark_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Saved summary: {summary_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Run PLAID retrieval benchmarks")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--samples", type=int, help="Override number of samples")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    run_plaid_benchmark(cfg, samples_override=args.samples)


if __name__ == "__main__":
    main()
