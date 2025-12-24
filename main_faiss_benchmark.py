"""
FAISS Index Benchmark Entry Point.

Runs comprehensive benchmarks comparing FAISS index types:
- Flat (baseline, exact search)
- IVF (inverted file, approximate search)
- IVF-PQ (product quantization, memory-efficient)

Usage:
    # Run full benchmark grid
    python main_faiss_benchmark.py --config configs/faiss_benchmark/full_grid.yml
    
    # Quick test with few samples
    python main_faiss_benchmark.py --config configs/faiss_benchmark/full_grid.yml --samples 5
    
    # Single index type
    python main_faiss_benchmark.py --config configs/faiss_benchmark/ivf_pq_8bit_m96.yml
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
from src.models.retriever import (
    load_embedding_model, embed_chunks, _compute_cache_key,
    save_embeddings, load_embeddings, create_query_from_document
)
from src.models.retriever_advanced import (
    build_flat_index, build_ivf_index, build_ivf_pq_index,
    compute_ground_truth_chunks, retrieve_and_benchmark,
    get_index_size_mb, set_nprobe, get_index_info,
    set_reproducibility_seed
)
from src.models.generator import concatenate_chunks, generate_from_chunks


def run_faiss_benchmark(cfg: Dict, samples_override: Optional[int] = None):
    """
    Run FAISS index benchmarks.
    
    Pipeline:
    1. Load documents and chunk them
    2. Embed chunks with MiniLM and BGE
    3. Build Flat/IVF/IVF-PQ indices
    4. Compute ground-truth using ROUGE-L with reference summary
    5. Benchmark retrieval (Recall@K, MRR, latency, index size)
    6. Generate summaries with top-K chunks
    7. Evaluate downstream ROUGE-L
    
    Args:
        cfg: Configuration dict from YAML
        samples_override: Override number of samples (for testing)
    """
    print("=" * 60)
    print("🔬 FAISS Index Benchmark")
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
    
    # Embedding models to compare
    embedding_models = cfg.get("embedding_models", [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-base-en-v1.5"
    ])
    
    # Summarization model (best performer from previous experiments)
    summarization_model = cfg.get("summarization_model", "google/bigbird-pegasus-large-arxiv")
    
    # Index configurations
    index_configs = cfg.get("index_configs", [
        {"type": "flat"},
        {"type": "ivf", "nlist": 1024, "nprobe": 8},
        {"type": "ivf", "nlist": 4096, "nprobe": 8},
        {"type": "ivf_pq", "nlist": 1024, "M": 48, "nbits": 8, "nprobe": 8},
        {"type": "ivf_pq", "nlist": 1024, "M": 96, "nbits": 8, "nprobe": 8},
        {"type": "ivf_pq", "nlist": 1024, "M": 48, "nbits": 4, "nprobe": 8},
        {"type": "ivf_pq", "nlist": 1024, "M": 96, "nbits": 4, "nprobe": 8},
    ])
    
    # nprobe values to test for IVF-based indices
    nprobe_values = cfg.get("nprobe_values", [4, 8, 16])
    
    # Output paths
    out_dir = cfg.get("out_dir", "results/faiss_benchmark")
    os.makedirs(out_dir, exist_ok=True)
    
    cache_dir = cfg.get("cache_dir", "cache/faiss_benchmark")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"\n📁 Dataset: {dataset_path}")
    print(f"📊 Samples: {samples}")
    print(f"🔧 Chunk size: {chunk_size}, Overlap: {overlap}, Method: {segmentation_method}")
    print(f"🎯 Top-K: {top_k}, Ground-truth top-N: {ground_truth_top_n}")
    print(f"📝 Summarization model: {summarization_model}")
    print(f"🔍 Embedding models: {embedding_models}")
    print(f"📦 Index configs: {len(index_configs)} types × {len(nprobe_values)} nprobe values")
    
    # Load data
    print(f"\n📥 Loading {samples} samples...")
    data = sample_records(dataset_path, samples, seed)
    
    # Load summarization model ONCE
    print(f"\n🤖 Loading summarization model: {summarization_model}...")
    base_tok = build_tokenizer(summarization_model)
    sum_model, sum_tok, sum_metadata = load_model(summarization_model, "default")
    print(f"  ✓ Model loaded (family: {sum_metadata.get('model_family', 'unknown')})")
    
    # Results storage
    all_results = []
    
    # Process each embedding model
    for emb_model_name in embedding_models:
        print(f"\n{'='*60}")
        print(f"🔄 Embedding Model: {emb_model_name}")
        print(f"{'='*60}")
        
        # Load embedding model ONCE
        emb_model = load_embedding_model(emb_model_name)
        print(f"  ✓ Loaded on {emb_model.device}")
        
        # Process each document
        for doc_idx, ex in enumerate(tqdm(data, desc="Documents")):
            doc = ex.get("text", "")
            ref = ex.get("ref", "")
            
            if not doc.strip() or not ref.strip():
                continue
            
            # Step 1: Chunk document
            chunks = chunk_text(doc, base_tok, chunk_size, overlap, method=segmentation_method)
            
            if len(chunks) < 2:
                continue
            
            # Step 2: Embed chunks (with caching)
            cache_key = _compute_cache_key(chunks, emb_model_name)
            embeddings = load_embeddings(cache_dir, cache_key)
            
            if embeddings is None:
                embeddings = embed_chunks(chunks, emb_model)
                save_embeddings(embeddings, cache_dir, cache_key)
            
            # Step 3: Compute ground-truth oracle
            gt_indices, gt_scores = compute_ground_truth_chunks(chunks, ref, ground_truth_top_n)
            
            # Step 4: Create query
            query_text = create_query_from_document(doc, base_tok, strategy=query_strategy)
            
            # Step 5: Benchmark each index configuration
            for idx_cfg in index_configs:
                idx_type = idx_cfg["type"]
                
                # Build index based on type
                # Note: embeddings from embed_chunks are already normalized
                # Index builders will normalize=True by default as fallback
                if idx_type == "flat":
                    index = build_flat_index(embeddings.copy(), normalize=True)
                    nprobe_list = [1]  # Flat doesn't use nprobe
                    
                elif idx_type == "ivf":
                    nlist = idx_cfg.get("nlist", 1024)
                    index = build_ivf_index(embeddings.copy(), nlist=nlist, nprobe=8, normalize=True, seed=seed)
                    nprobe_list = nprobe_values
                    
                elif idx_type == "ivf_pq":
                    nlist = idx_cfg.get("nlist", 1024)
                    M = idx_cfg.get("M", 96)
                    nbits = idx_cfg.get("nbits", 8)
                    index = build_ivf_pq_index(embeddings.copy(), nlist=nlist, M=M, nbits=nbits, nprobe=8, normalize=True, seed=seed)
                    nprobe_list = nprobe_values
                else:
                    continue
                
                # Get index info
                index_info = get_index_info(index)
                index_size_mb = index_info["size_mb"]
                
                # Test different nprobe values
                for nprobe in nprobe_list:
                    if idx_type != "flat":
                        set_nprobe(index, nprobe)
                    
                    # Retrieve and benchmark
                    bench_results = retrieve_and_benchmark(
                        query_text, chunks, index, emb_model,
                        top_k, gt_indices
                    )
                    
                    # Get top-K chunks for summary generation
                    retrieved_indices = bench_results["retrieved_indices"][:top_k]
                    
                    # Generate summary
                    t0 = start_prof()
                    context_text = concatenate_chunks(chunks, retrieved_indices, base_tok, max_tokens=16000, preserve_order=True)
                    pred = generate_from_chunks(sum_model, sum_tok, context_text, cfg.get("gen_max_tokens", 512))
                    lat, mem = stop_prof(t0)
                    
                    # Evaluate downstream ROUGE-L
                    mets = text_metrics([pred], [ref])
                    
                    # Build result row
                    result = {
                        "doc_idx": doc_idx,
                        "embedding_model": emb_model_name,
                        "index_type": idx_type,
                        "nlist": idx_cfg.get("nlist", 0),
                        "M": idx_cfg.get("M", 0),
                        "nbits": idx_cfg.get("nbits", 0),
                        "nprobe": nprobe if idx_type != "flat" else 0,
                        "num_chunks": len(chunks),
                        "gt_top_n": ground_truth_top_n,
                        "top_k": top_k,
                        "gt_valid": bench_results.get("gt_valid", True),
                        "recall_at_1": bench_results["recall_at_1"],
                        "recall_at_5": bench_results["recall_at_5"],
                        "recall_at_10": bench_results["recall_at_10"],
                        "mrr": bench_results["mrr"],
                        "retrieval_latency_ms": bench_results["latency_ms"],
                        "index_size_mb": index_size_mb,
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
    
    # Report gt_valid statistics
    if "gt_valid" in df.columns:
        gt_valid_count = df["gt_valid"].sum()
        gt_invalid_count = len(df) - gt_valid_count
        print(f"📊 Ground-truth validity: {gt_valid_count}/{len(df)} valid ({gt_invalid_count} excluded from recall/MRR)")
    
    # Aggregated results (mean per configuration)
    agg_columns = ["embedding_model", "index_type", "nlist", "M", "nbits", "nprobe"]
    metric_columns = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr", 
                     "retrieval_latency_ms", "index_size_mb", 
                     "downstream_rouge_l", "downstream_bertscore"]
    
    agg_df = df.groupby(agg_columns)[metric_columns].mean().reset_index()
    agg_df = agg_df.round(4)
    
    agg_path = os.path.join(out_dir, "benchmark_aggregated.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"✅ Saved aggregated results: {agg_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY (Aggregated)")
    print("=" * 80)
    print(agg_df.to_string(index=False))
    
    # Save summary JSON
    summary = {
        "config": cfg,
        "num_samples": len(data),
        "embedding_models": embedding_models,
        "index_configs": index_configs,
        "nprobe_values": nprobe_values,
        "gt_valid_count": int(gt_valid_count) if "gt_valid" in df.columns else len(df),
        "gt_invalid_count": int(gt_invalid_count) if "gt_valid" in df.columns else 0,
        "aggregated_results": agg_df.to_dict(orient="records")
    }
    
    summary_path = os.path.join(out_dir, "benchmark_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary: {summary_path}")
    
    return df, agg_df


def main():
    parser = argparse.ArgumentParser(description="Run FAISS index benchmarks")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--samples", type=int, help="Override number of samples (for testing)")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    run_faiss_benchmark(cfg, samples_override=args.samples)


if __name__ == "__main__":
    main()
