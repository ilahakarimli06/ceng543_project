"""
FAISS Index Benchmark v2 - Global Index Architecture.

Key improvements over v1:
- Global index: All chunks indexed together (enables proper IVF/IVF-PQ)
- doc_id metadata: Each chunk knows which document it belongs to
- Two-stage retrieval: Optional cross-encoder reranking
- Configurable chunk_size and overlap

Usage:
    python main_faiss_benchmark_v2.py --config configs/faiss_benchmark/full_grid_v2.yml
"""

import yaml
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict

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
    set_reproducibility_seed, _normalize_embeddings, _ensure_contiguous_float32
)
from src.models.generator import concatenate_chunks, generate_from_chunks

# Optional: Cross-encoder for reranking
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


def chunk_all_documents(
    documents: List[Dict],
    tokenizer,
    chunk_size: int,
    overlap: int,
    method: str = "fixed"
) -> Tuple[List[Dict], Dict[int, List[int]]]:
    """
    Chunk all documents and build global chunk list with metadata.
    
    Returns:
        all_chunks: List of chunk dicts with doc_id, chunk_id, text
        doc_to_chunks: Mapping from doc_id to list of global chunk indices
    """
    all_chunks = []
    doc_to_chunks = defaultdict(list)
    
    for doc_id, doc in enumerate(tqdm(documents, desc="Chunking documents")):
        text = doc.get("text", "")
        if not text.strip():
            continue
            
        chunks = chunk_text(text, tokenizer, chunk_size, overlap, method=method)
        
        for local_chunk_id, chunk in enumerate(chunks):
            global_chunk_id = len(all_chunks)
            
            # Add metadata
            chunk["doc_id"] = doc_id
            chunk["chunk_id"] = local_chunk_id
            chunk["global_id"] = global_chunk_id
            
            all_chunks.append(chunk)
            doc_to_chunks[doc_id].append(global_chunk_id)
    
    return all_chunks, dict(doc_to_chunks)


def rerank_with_cross_encoder(
    query: str,
    chunks: List[Dict],
    retrieved_indices: List[int],
    cross_encoder,
    top_n: int = 10
) -> List[int]:
    """
    Rerank retrieved chunks using cross-encoder.
    
    Args:
        query: Query text
        chunks: All chunks
        retrieved_indices: Indices from first-stage retrieval
        cross_encoder: CrossEncoder model
        top_n: Number of results after reranking
        
    Returns:
        Reranked top_n indices
    """
    if not retrieved_indices:
        return []
    
    # Create query-chunk pairs
    pairs = [(query, chunks[idx]["text"]) for idx in retrieved_indices]
    
    # Score with cross-encoder
    scores = cross_encoder.predict(pairs)
    
    # Sort by score (descending)
    scored_indices = list(zip(retrieved_indices, scores))
    scored_indices.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_n
    return [idx for idx, _ in scored_indices[:top_n]]


def retrieve_for_document(
    doc_id: int,
    query_text: str,
    all_chunks: List[Dict],
    doc_to_chunks: Dict[int, List[int]],
    index,
    embedding_model,
    k_coarse: int,
    k_final: int,
    cross_encoder=None
) -> Dict:
    """
    Retrieve chunks for a single document query.
    
    Two-stage retrieval:
    1. FAISS retrieval (k_coarse candidates)
    2. Optional cross-encoder rerank (k_final results)
    """
    import faiss
    
    # Embed query
    try:
        query_emb = embedding_model.encode(
            [query_text],
            convert_to_numpy=True,
            show_progress_bar=False
        )
    except TypeError:
        query_emb = np.array(embedding_model.encode([query_text], show_progress_bar=False))
    
    query_emb = _ensure_contiguous_float32(query_emb)
    faiss.normalize_L2(query_emb)
    
    # Stage 1: FAISS retrieval
    start = time.perf_counter()
    scores, indices = index.search(query_emb, k_coarse)
    faiss_latency_ms = (time.perf_counter() - start) * 1000
    
    # Filter valid indices
    retrieved_indices = [int(idx) for idx in indices[0] if 0 <= idx < len(all_chunks)]
    retrieved_scores = [float(s) for s in scores[0][:len(retrieved_indices)]]
    
    # Stage 2: Optional reranking
    rerank_latency_ms = 0.0
    if cross_encoder is not None and len(retrieved_indices) > k_final:
        start = time.perf_counter()
        retrieved_indices = rerank_with_cross_encoder(
            query_text, all_chunks, retrieved_indices, cross_encoder, k_final
        )
        rerank_latency_ms = (time.perf_counter() - start) * 1000
    else:
        retrieved_indices = retrieved_indices[:k_final]
    
    # Get chunks for this document's ground-truth
    doc_chunk_indices = doc_to_chunks.get(doc_id, [])
    
    return {
        "retrieved_indices": retrieved_indices,
        "retrieved_scores": retrieved_scores[:len(retrieved_indices)],
        "doc_chunk_indices": doc_chunk_indices,
        "faiss_latency_ms": faiss_latency_ms,
        "rerank_latency_ms": rerank_latency_ms,
        "total_latency_ms": faiss_latency_ms + rerank_latency_ms
    }


def run_faiss_benchmark_v2(cfg: Dict, samples_override: Optional[int] = None):
    """
    Run FAISS index benchmarks with global index architecture.
    
    Key differences from v1:
    - All chunks indexed together (global index)
    - Each chunk has doc_id, chunk_id metadata
    - Optional two-stage retrieval with cross-encoder reranking
    """
    print("=" * 70)
    print("🔬 FAISS Index Benchmark v2 - Global Index Architecture")
    print("=" * 70)
    
    # Extract config
    samples = samples_override or cfg.get("samples", 60)
    seed = cfg.get("seed", 42)
    dataset_path = cfg["dataset_path"]
    
    # Chunking config (updated defaults)
    chunk_size = cfg.get("chunk_size", 384)
    overlap = cfg.get("overlap", 128)
    segmentation_method = cfg.get("segmentation_method", "fixed")
    
    # Retrieval config
    k_coarse = cfg.get("k_coarse", 100)  # First-stage retrieval
    k_final = cfg.get("k_final", 10)     # After reranking
    ground_truth_top_n = cfg.get("ground_truth_top_n", 5)
    query_strategy = cfg.get("query_strategy", "first_tokens")
    
    # Reranking config
    use_reranking = cfg.get("use_reranking", False)
    reranker_model = cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Embedding models
    embedding_models = cfg.get("embedding_models", [
        "sentence-transformers/all-MiniLM-L6-v2"
    ])
    
    # Summarization model
    summarization_model = cfg.get("summarization_model", "google/bigbird-pegasus-large-arxiv")
    
    # Index configurations
    index_configs = cfg.get("index_configs", [
        {"type": "flat"},
        {"type": "ivf", "nlist": 256},
        {"type": "ivf_pq", "nlist": 256, "M": 48, "nbits": 8}
    ])
    
    nprobe_values = cfg.get("nprobe_values", [8, 16, 32])
    
    # Output paths
    out_dir = cfg.get("out_dir", "results/faiss_benchmark_v2")
    os.makedirs(out_dir, exist_ok=True)
    
    cache_dir = cfg.get("cache_dir", "cache/faiss_benchmark_v2")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Print config
    print(f"\n📁 Dataset: {dataset_path}")
    print(f"📊 Samples: {samples}")
    print(f"🔧 Chunk size: {chunk_size}, Overlap: {overlap}")
    print(f"🎯 k_coarse: {k_coarse}, k_final: {k_final}")
    print(f"🔄 Reranking: {'✓ ' + reranker_model if use_reranking else '✗ disabled'}")
    print(f"📝 Summarization: {summarization_model}")
    print(f"🔍 Embedding models: {embedding_models}")
    print(f"📦 Index configs: {len(index_configs)} types × {len(nprobe_values)} nprobe")
    
    # Set seed
    set_reproducibility_seed(seed)
    
    # Load data
    print(f"\n📥 Loading {samples} samples...")
    data = sample_records(dataset_path, samples, seed)
    
    # Build tokenizer
    base_tok = build_tokenizer(summarization_model)
    
    # Chunk ALL documents (global chunk list)
    print(f"\n📦 Chunking all documents (size={chunk_size}, overlap={overlap})...")
    all_chunks, doc_to_chunks = chunk_all_documents(
        data, base_tok, chunk_size, overlap, segmentation_method
    )
    print(f"   ✓ Total chunks: {len(all_chunks)} from {len(data)} documents")
    print(f"   ✓ Avg chunks/doc: {len(all_chunks) / len(data):.1f}")
    
    # Load summarization model ONCE
    print(f"\n🤖 Loading summarization model: {summarization_model}...")
    sum_model, sum_tok, sum_metadata = load_model(summarization_model, "default")
    print(f"   ✓ Model loaded (family: {sum_metadata.get('model_family', 'unknown')})")
    
    # Load cross-encoder if enabled
    cross_encoder = None
    if use_reranking:
        if CROSS_ENCODER_AVAILABLE:
            print(f"\n🔄 Loading cross-encoder: {reranker_model}...")
            cross_encoder = CrossEncoder(reranker_model)
            print("   ✓ Cross-encoder loaded")
        else:
            print("\n⚠️ Cross-encoder not available (sentence-transformers not installed)")
            use_reranking = False
    
    # Results storage
    all_results = []
    
    # Process each embedding model
    for emb_model_name in embedding_models:
        print(f"\n{'='*70}")
        print(f"🔄 Embedding Model: {emb_model_name}")
        print(f"{'='*70}")
        
        # Load embedding model
        emb_model = load_embedding_model(emb_model_name)
        print(f"   ✓ Loaded on {emb_model.device}")
        
        # Embed ALL chunks (global embeddings)
        print(f"\n📊 Embedding {len(all_chunks)} chunks...")
        cache_key = _compute_cache_key(all_chunks, emb_model_name)
        embeddings = load_embeddings(cache_dir, cache_key)
        
        if embeddings is None:
            embeddings = embed_chunks(all_chunks, emb_model)
            save_embeddings(embeddings, cache_dir, cache_key)
            print(f"   ✓ Embedded and cached: {embeddings.shape}")
        else:
            print(f"   ✓ Loaded from cache: {embeddings.shape}")
        
        # Compute ground-truth for each document
        print("\n🎯 Computing ground-truth (ROUGE-L oracle)...")
        doc_ground_truths = {}
        for doc_id, doc in enumerate(tqdm(data, desc="Ground-truth")):
            ref = doc.get("ref", "")
            doc_chunk_ids = doc_to_chunks.get(doc_id, [])
            
            if not doc_chunk_ids or not ref.strip():
                doc_ground_truths[doc_id] = []
                continue
            
            # Get chunks for this document
            doc_chunks = [all_chunks[i] for i in doc_chunk_ids]
            
            # Compute ground-truth indices (local to document)
            gt_local, _ = compute_ground_truth_chunks(doc_chunks, ref, ground_truth_top_n)
            
            # Convert to global indices
            gt_global = [doc_chunk_ids[i] for i in gt_local]
            doc_ground_truths[doc_id] = gt_global
        
        # Test each index configuration
        for idx_cfg in index_configs:
            idx_type = idx_cfg["type"]
            
            print(f"\n📦 Building {idx_type.upper()} index...")
            
            # Build index on normalized copy
            emb_copy = embeddings.copy()
            
            if idx_type == "flat":
                index = build_flat_index(emb_copy, normalize=True)
                nprobe_list = [1]  # Flat doesn't use nprobe
            elif idx_type == "ivf":
                nlist = idx_cfg.get("nlist", 256)
                index = build_ivf_index(emb_copy, nlist=nlist, nprobe=8, normalize=True, seed=seed)
                nprobe_list = nprobe_values
            elif idx_type == "ivf_pq":
                nlist = idx_cfg.get("nlist", 256)
                M = idx_cfg.get("M", 48)
                nbits = idx_cfg.get("nbits", 8)
                index = build_ivf_pq_index(emb_copy, nlist=nlist, M=M, nbits=nbits, normalize=True, seed=seed)
                nprobe_list = nprobe_values
            else:
                continue
            
            # Get index info
            index_info = get_index_info(index)
            index_size_mb = index_info["size_mb"]
            print(f"   ✓ Index built: {index_info['ntotal']} vectors, {index_size_mb:.2f} MB")
            
            # Test different nprobe values
            for nprobe in nprobe_list:
                if idx_type != "flat":
                    set_nprobe(index, nprobe)
                
                print(f"\n   Testing nprobe={nprobe}...")
                
                # Benchmark each document
                for doc_id, doc in enumerate(tqdm(data, desc=f"   {idx_type}/nprobe={nprobe}")):
                    ref = doc.get("ref", "")
                    text = doc.get("text", "")
                    
                    if not ref.strip() or not text.strip():
                        continue
                    
                    # Create query
                    query_text = create_query_from_document(text, base_tok, strategy=query_strategy)
                    
                    # Retrieve
                    ret_result = retrieve_for_document(
                        doc_id, query_text, all_chunks, doc_to_chunks,
                        index, emb_model, k_coarse, k_final, cross_encoder
                    )
                    
                    retrieved_indices = ret_result["retrieved_indices"]
                    gt_indices = doc_ground_truths.get(doc_id, [])
                    
                    # Compute retrieval metrics
                    from src.models.retriever_advanced import compute_recall_at_k, compute_mrr
                    recall_1 = compute_recall_at_k(retrieved_indices, gt_indices, 1) or 0.0
                    recall_5 = compute_recall_at_k(retrieved_indices, gt_indices, 5) or 0.0
                    recall_10 = compute_recall_at_k(retrieved_indices, gt_indices, 10) or 0.0
                    mrr = compute_mrr(retrieved_indices, gt_indices) or 0.0
                    
                    # Generate summary from retrieved chunks
                    retrieved_chunks = [all_chunks[i] for i in retrieved_indices if i < len(all_chunks)]
                    
                    t0 = start_prof()
                    context_text = concatenate_chunks(
                        retrieved_chunks, list(range(len(retrieved_chunks))),
                        base_tok, max_tokens=16000, preserve_order=True
                    )
                    pred = generate_from_chunks(sum_model, sum_tok, context_text, cfg.get("gen_max_tokens", 512))
                    lat, mem = stop_prof(t0)
                    
                    # Evaluate downstream
                    mets = text_metrics([pred], [ref])
                    
                    # Build result row
                    result = {
                        "doc_id": doc_id,
                        "embedding_model": emb_model_name,
                        "index_type": idx_type,
                        "nlist": idx_cfg.get("nlist", 0),
                        "M": idx_cfg.get("M", 0),
                        "nbits": idx_cfg.get("nbits", 0),
                        "nprobe": nprobe if idx_type != "flat" else 0,
                        "k_coarse": k_coarse,
                        "k_final": k_final,
                        "use_reranking": use_reranking,
                        "num_chunks_total": len(all_chunks),
                        "num_chunks_doc": len(doc_to_chunks.get(doc_id, [])),
                        "recall_at_1": recall_1,
                        "recall_at_5": recall_5,
                        "recall_at_10": recall_10,
                        "mrr": mrr,
                        "faiss_latency_ms": ret_result["faiss_latency_ms"],
                        "rerank_latency_ms": ret_result["rerank_latency_ms"],
                        "total_retrieval_ms": ret_result["total_latency_ms"],
                        "index_size_mb": index_size_mb,
                        "downstream_rouge_l": mets["rougeL"],
                        "downstream_bertscore": mets["bertscore_f1"],
                        "summarization_latency_s": lat,
                        "gpu_peak_gb": mem
                    }
                    
                    all_results.append(result)
    
    # Save results
    df = pd.DataFrame(all_results)
    
    # Per-document results
    out_path = os.path.join(out_dir, "benchmark_full.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved per-document results: {out_path}")
    
    # Aggregated results
    agg_columns = ["embedding_model", "index_type", "nlist", "M", "nbits", "nprobe", "use_reranking"]
    metric_columns = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr",
                     "faiss_latency_ms", "rerank_latency_ms", "total_retrieval_ms",
                     "index_size_mb", "downstream_rouge_l", "downstream_bertscore"]
    
    agg_df = df.groupby(agg_columns)[metric_columns].mean().reset_index()
    agg_df = agg_df.round(4)
    
    agg_path = os.path.join(out_dir, "benchmark_aggregated.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"✅ Saved aggregated results: {agg_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY (Aggregated)")
    print("=" * 80)
    print(agg_df.to_string(index=False))
    
    # Save summary JSON
    summary = {
        "config": cfg,
        "num_samples": len(data),
        "total_chunks": len(all_chunks),
        "avg_chunks_per_doc": len(all_chunks) / len(data),
        "embedding_models": embedding_models,
        "index_configs": index_configs,
        "aggregated_results": agg_df.to_dict(orient="records")
    }
    
    summary_path = os.path.join(out_dir, "benchmark_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary: {summary_path}")
    
    return df, agg_df


def main():
    parser = argparse.ArgumentParser(description="FAISS Benchmark v2 - Global Index")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--samples", type=int, help="Override number of samples")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    run_faiss_benchmark_v2(cfg, samples_override=args.samples)


if __name__ == "__main__":
    main()
