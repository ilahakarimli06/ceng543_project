"""
Main entry point for chunk retrieval experiments.

Usage:
    # LED model
    python main_chunk_retrieval.py --config configs/chunk_retrieval/arxiv_extra_long_c1024_ov0_k3_fixed.yml
    
    # LongT5 model
    python main_chunk_retrieval.py --config configs/chunk_retrieval/arxiv_extra_long_c1024_ov0_k3_fixed.yml --model google/long-t5-local-base
"""

import yaml
import torch
import argparse
import json
import os
import pandas as pd

from src.utils.io import sample_records
from src.utils.chunker import chunk_text
from src.utils.window import build_tokenizer
from src.utils.prof import start_prof, stop_prof
from src.eval.metric import text_metrics
from src.models.sliding import load_model
from src.models.retriever import (load_embedding_model, embed_chunks, build_faiss_index, 
                                   retrieve_top_k, create_query_from_document,
                                   save_embeddings, load_embeddings, save_index, load_index,
                                   _compute_cache_key)
from src.models.generator import concatenate_chunks, generate_from_chunks


def run_chunk_retrieval(cfg, attention_override=None, model_override=None):
    """
    Run chunk retrieval experiments.
    
    Pipeline:
    1. Chunk document using specified method
    2. Embed chunks with sentence-transformers
    3. Build FAISS index
    4. Retrieve top-K chunks
    5. Concatenate chunks (respecting 16k token budget)
    6. Generate summary with LED/LongT5
    7. Evaluate with ROUGE-L and BERTScore
    """
    # Override attention_impl if specified via command line
    if attention_override:
        cfg["attention_impl"] = attention_override
    
    # Override model_name if specified via command line
    if model_override:
        cfg["model_name"] = model_override
    
    rows = []
    samples_data = []  # Store summaries for JSON output
    
    # Setup paths
    out_path = cfg.get("out_csv", "results/chunk_retrieval/output.csv")
    if attention_override and attention_override != "default":
        out_path = out_path.replace(".csv", f"_{attention_override}.csv")
    if model_override:
        model_suffix = "longt5" if "long-t5" in model_override.lower() or "longt5" in model_override.lower() else "led"
        out_path = out_path.replace(".csv", f"_{model_suffix}.csv")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Load data
    data = sample_records(cfg["dataset_path"], cfg["samples"], cfg.get("seed", 42))
    
    # Load summarization model (LED or LongT5)
    base_tok = build_tokenizer(cfg["model_name"])
    model, tok, metadata = load_model(cfg["model_name"], cfg.get("attention_impl", "default"))
    
    # Extract config parameters
    chunk_size = cfg["chunk_size"]
    overlap = cfg["overlap"]
    segmentation_method = cfg.get("segmentation_method", "fixed")
    top_k = cfg["top_k"]
    embedding_model_name = cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    query_strategy = cfg.get("query_strategy", "first_tokens")
    
    print(f"🔧 Config: chunk_size={chunk_size}, overlap={overlap}, method={segmentation_method}, top_k={top_k}")
    print(f"🔧 Embedding model: {embedding_model_name}")
    print(f"🔧 Query strategy: {query_strategy}")
    
    # Load embedding model ONCE (critical for performance)
    print(f"\n🔄 Loading embedding model: {embedding_model_name}...")
    embedding_model = load_embedding_model(embedding_model_name)
    print(f"  ✓ Model loaded on {embedding_model.device}")
    
    # Setup cache directory
    cache_dir = cfg.get("cache_dir", "cache/embeddings")
    use_cache = cfg.get("use_cache", True)
    print(f"\n💾 Cache: {'enabled' if use_cache else 'disabled'} (dir: {cache_dir})")
    
    for idx, ex in enumerate(data):
        doc = ex.get("text", "")
        ref = ex.get("ref", "")
        
        print(f"\n📄 Processing document {idx + 1}/{len(data)}...")
        
        t0 = start_prof()
        
        # Step 1: Chunk document
        chunks = chunk_text(doc, base_tok, chunk_size, overlap, method=segmentation_method)
        print(f"  ✓ Created {len(chunks)} chunks")
        
        # Step 2: Embed chunks (using pre-loaded model with caching)
        cache_key = _compute_cache_key(chunks, embedding_model_name) if use_cache else None
        
        if use_cache:
            # Try to load from cache
            embeddings = load_embeddings(cache_dir, cache_key)
            if embeddings is not None:
                print(f"  ✓ Loaded embeddings from cache (dim={embeddings.shape[1]})")
            else:
                # Compute and save to cache
                embeddings = embed_chunks(chunks, embedding_model)
                save_embeddings(embeddings, cache_dir, cache_key)
                print(f"  ✓ Embedded chunks (dim={embeddings.shape[1]}) [cached]")
        else:
            embeddings = embed_chunks(chunks, embedding_model)
            print(f"  ✓ Embedded chunks (dim={embeddings.shape[1]})")
        
        # Step 3: Build FAISS index (with caching)
        if use_cache:
            # Try to load from cache
            index = load_index(cache_dir, cache_key)
            if index is not None:
                print(f"  ✓ Loaded FAISS index from cache")
            else:
                # Build and save to cache
                index = build_faiss_index(embeddings)
                save_index(index, cache_dir, cache_key)
                print(f"  ✓ Built FAISS index [cached]")
        else:
            index = build_faiss_index(embeddings)
            print(f"  ✓ Built FAISS index")
        
        # Step 4: Create query and retrieve top-K chunks
        query_text = create_query_from_document(doc, base_tok, strategy=query_strategy)
        top_k_indices, top_k_scores = retrieve_top_k(
            query_text, chunks, index, top_k, embedding_model
        )
        print(f"  ✓ Retrieved top-{top_k} chunks (scores: {[f'{s:.3f}' for s in top_k_scores[:3]]}...)")
        
        # Step 5: Concatenate chunks
        context_text = concatenate_chunks(chunks, top_k_indices, base_tok, max_tokens=16000, preserve_order=True)
        context_token_count = len(base_tok(context_text, add_special_tokens=False)["input_ids"])
        print(f"  ✓ Concatenated context ({context_token_count} tokens)")
        
        # Step 6: Generate summary
        pred = generate_from_chunks(model, tok, context_text, cfg["gen_max_tokens"])
        print(f"  ✓ Generated summary ({len(pred)} chars)")
        
        # Step 7: Evaluate
        lat, mem = stop_prof(t0)
        mets = text_metrics([pred], [ref])
        throughput = 3600 / lat if lat > 0 else 0  # docs per hour
        
        print(f"  ✓ Metrics: ROUGE-L={mets['rougeL']:.4f}, BERTScore={mets['bertscore_f1']:.4f}")
        print(f"  ✓ Performance: {lat:.2f}s, {mem:.2f}GB, {throughput:.1f} docs/hour")
        
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
            "chunk_size": chunk_size,
            "overlap": overlap,
            "segmentation_method": segmentation_method,
            "top_k": top_k,
            "embedding_model": embedding_model_name,
            "query_strategy": query_strategy,
            "num_chunks": len(chunks),
            "context_tokens": context_token_count,
            "latency": lat,
            "gpu_peak_gb": mem,
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
            "num_chunks": len(chunks),
            "retrieved_chunk_indices": top_k_indices,
            "retrieved_chunk_scores": top_k_scores,
            "context_tokens": context_token_count,
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
    print(f"\n✅ Results saved: {out_path}")
    
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
            "chunk_size": chunk_size,
            "overlap": overlap,
            "segmentation_method": segmentation_method,
            "top_k": top_k,
            "embedding_model": embedding_model_name,
            "query_strategy": query_strategy,
            "category": cfg.get("category", "unknown")
        },
        "summary_stats": {
            "total_samples": len(samples_data),
            "avg_rougeL": df["rougeL"].mean(),
            "avg_bertscore": df["bertscore_f1"].mean(),
            "min_rougeL": df["rougeL"].min(),
            "min_bertscore": df["bertscore_f1"].min(),
            "avg_num_chunks": df["num_chunks"].mean(),
            "avg_context_tokens": df["context_tokens"].mean()
        },
        "samples": selected_samples
    }
    
    with open(samples_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Samples saved: {samples_path} ({len(selected_samples)} examples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run chunk retrieval experiments")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--attention", choices=["default", "flash_attention_2"], 
                       help="Override attention implementation (optional)")
    parser.add_argument("--model", 
                       help="Override model name (e.g., google/long-t5-local-base)")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    run_chunk_retrieval(cfg, attention_override=args.attention, model_override=args.model)
