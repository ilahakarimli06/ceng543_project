"""
Equal Budget Comparison Report Generator.

Compares all retrieval methods (Dense, ColBERT, PLAID) at equal index size budgets.
Reads results from all benchmark runs and generates comparison tables.

Usage:
    # Generate comparison at ~500MB budget
    python generate_equal_budget_report.py --budget 500
    
    # Generate comparison for multiple budgets
    python generate_equal_budget_report.py --budgets 100 500 1000
    
    # Specify custom result directories
    python generate_equal_budget_report.py --faiss-dir results/faiss_benchmark \
        --colbert-dir results/colbert_benchmark --plaid-dir results/plaid_benchmark
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime


def load_faiss_results(results_dir: str) -> Optional[pd.DataFrame]:
    """
    Load FAISS benchmark results.
    
    Args:
        results_dir: Path to FAISS results directory
        
    Returns:
        DataFrame with FAISS results or None if not found
    """
    # Try aggregated first
    agg_path = os.path.join(results_dir, "benchmark_aggregated.csv")
    if os.path.exists(agg_path):
        df = pd.read_csv(agg_path)
        df["method"] = "Dense"
        df["method_detail"] = df.apply(
            lambda r: f"{r['index_type']}" + 
                     (f"_M{int(r['M'])}" if r.get('M', 0) > 0 else "") +
                     (f"_nlist{int(r['nlist'])}" if r.get('nlist', 0) > 0 else ""),
            axis=1
        )
        return df
    
    # Try full results
    full_path = os.path.join(results_dir, "benchmark_full.csv")
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        df["method"] = "Dense"
        return df
    
    print(f"  ⚠️ No FAISS results found in {results_dir}")
    return None


def load_colbert_results(results_dir: str) -> Optional[pd.DataFrame]:
    """
    Load ColBERT benchmark results.
    
    Args:
        results_dir: Path to ColBERT results directory
        
    Returns:
        DataFrame with ColBERT results or None if not found
    """
    full_path = os.path.join(results_dir, "benchmark_full.csv")
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        df["method"] = "ColBERT"
        df["method_detail"] = "ColBERT-Full"
        df["index_type"] = "colbert"
        return df
    
    summary_path = os.path.join(results_dir, "benchmark_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
        if "per_document" in data:
            df = pd.DataFrame(data["per_document"])
            df["method"] = "ColBERT"
            df["method_detail"] = "ColBERT-Full"
            return df
    
    print(f"  ⚠️ No ColBERT results found in {results_dir}")
    return None


def load_plaid_results(results_dir: str) -> Optional[pd.DataFrame]:
    """
    Load PLAID benchmark results.
    
    Args:
        results_dir: Path to PLAID results directory
        
    Returns:
        DataFrame with PLAID results or None if not found
    """
    # Try aggregated first
    agg_path = os.path.join(results_dir, "benchmark_aggregated.csv")
    if os.path.exists(agg_path):
        df = pd.read_csv(agg_path)
        df["method"] = "PLAID"
        df["method_detail"] = df.apply(
            lambda r: f"PLAID-T{int(r['tokens_per_chunk'])}_M{int(r['M'])}_b{int(r['nbits'])}",
            axis=1
        )
        df["index_type"] = "plaid"
        # Rename for consistency
        if "plaid_index_size_mb" in df.columns:
            df["index_size_mb"] = df["plaid_index_size_mb"]
        return df
    
    # Try full results
    full_path = os.path.join(results_dir, "benchmark_full.csv")
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        df["method"] = "PLAID"
        df["method_detail"] = df.apply(
            lambda r: f"PLAID-T{int(r['tokens_per_chunk'])}_M{int(r['M'])}_b{int(r['nbits'])}",
            axis=1
        )
        if "plaid_index_size_mb" in df.columns:
            df["index_size_mb"] = df["plaid_index_size_mb"]
        return df
    
    print(f"  ⚠️ No PLAID results found in {results_dir}")
    return None


def find_nearest_budget_config(
    df: pd.DataFrame, 
    target_budget_mb: float,
    size_column: str = "index_size_mb"
) -> pd.DataFrame:
    """
    Find configurations closest to target budget.
    
    Args:
        df: Results DataFrame
        target_budget_mb: Target index size in MB
        size_column: Column containing index size
        
    Returns:
        Filtered DataFrame with configs nearest to budget
    """
    if size_column not in df.columns:
        return df
    
    # For each unique method_detail, find the one closest to budget
    if "method_detail" in df.columns:
        best_configs = []
        for method_detail in df["method_detail"].unique():
            method_df = df[df["method_detail"] == method_detail]
            
            # Get average size for this config
            avg_size = method_df[size_column].mean()
            
            # Store with distance to target
            best_configs.append({
                "method_detail": method_detail,
                "avg_size": avg_size,
                "distance": abs(avg_size - target_budget_mb),
                "df": method_df
            })
        
        # Sort by distance to target
        best_configs.sort(key=lambda x: x["distance"])
        
        # Take configs within 50% of target
        tolerance = target_budget_mb * 0.5
        filtered = [c for c in best_configs if c["distance"] <= tolerance]
        
        if filtered:
            return pd.concat([c["df"] for c in filtered], ignore_index=True)
    
    # Fallback: filter by size range
    lower = target_budget_mb * 0.5
    upper = target_budget_mb * 1.5
    return df[(df[size_column] >= lower) & (df[size_column] <= upper)]


def generate_comparison_table(
    faiss_df: Optional[pd.DataFrame],
    colbert_df: Optional[pd.DataFrame],
    plaid_df: Optional[pd.DataFrame],
    budget_mb: float
) -> pd.DataFrame:
    """
    Generate comparison table at given budget.
    
    Args:
        faiss_df: FAISS results
        colbert_df: ColBERT results
        plaid_df: PLAID results
        budget_mb: Target index size in MB
        
    Returns:
        Comparison DataFrame
    """
    comparison_rows = []
    
    # Metrics to compare
    metrics = [
        "recall_at_5", "recall_at_10", "mrr",
        "retrieval_latency_ms", "index_size_mb",
        "downstream_rouge_l", "downstream_bertscore"
    ]
    
    # Process FAISS
    if faiss_df is not None and len(faiss_df) > 0:
        filtered = find_nearest_budget_config(faiss_df, budget_mb)
        if len(filtered) > 0:
            for method_detail in filtered["method_detail"].unique():
                method_rows = filtered[filtered["method_detail"] == method_detail]
                row = {
                    "Method": "Dense",
                    "Config": method_detail,
                }
                for m in metrics:
                    if m in method_rows.columns:
                        row[m] = method_rows[m].mean()
                comparison_rows.append(row)
    
    # Process ColBERT
    if colbert_df is not None and len(colbert_df) > 0:
        row = {
            "Method": "ColBERT",
            "Config": "ColBERT-Full",
        }
        for m in metrics:
            if m in colbert_df.columns:
                row[m] = colbert_df[m].mean()
        comparison_rows.append(row)
    
    # Process PLAID
    if plaid_df is not None and len(plaid_df) > 0:
        filtered = find_nearest_budget_config(plaid_df, budget_mb)
        if len(filtered) > 0:
            for method_detail in filtered["method_detail"].unique():
                method_rows = filtered[filtered["method_detail"] == method_detail]
                row = {
                    "Method": "PLAID",
                    "Config": method_detail,
                }
                for m in metrics:
                    if m in method_rows.columns:
                        row[m] = method_rows[m].mean()
                comparison_rows.append(row)
    
    if not comparison_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(comparison_rows)
    
    # Round numeric columns
    for col in df.columns:
        if col not in ["Method", "Config"] and df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].round(4)
    
    return df


def generate_tradeoff_analysis(comparison_df: pd.DataFrame) -> str:
    """
    Generate text analysis of tradeoffs.
    
    Args:
        comparison_df: Comparison table
        
    Returns:
        Analysis text
    """
    if len(comparison_df) == 0:
        return "No data available for analysis."
    
    analysis = []
    
    # Best by recall
    if "recall_at_10" in comparison_df.columns:
        best_recall = comparison_df.loc[comparison_df["recall_at_10"].idxmax()]
        analysis.append(f"**Best Recall@10**: {best_recall['Config']} ({best_recall['recall_at_10']:.4f})")
    
    # Best by latency
    if "retrieval_latency_ms" in comparison_df.columns:
        best_latency = comparison_df.loc[comparison_df["retrieval_latency_ms"].idxmin()]
        analysis.append(f"**Fastest**: {best_latency['Config']} ({best_latency['retrieval_latency_ms']:.2f}ms)")
    
    # Best by ROUGE-L
    if "downstream_rouge_l" in comparison_df.columns:
        best_rouge = comparison_df.loc[comparison_df["downstream_rouge_l"].idxmax()]
        analysis.append(f"**Best ROUGE-L**: {best_rouge['Config']} ({best_rouge['downstream_rouge_l']:.4f})")
    
    # Smallest index
    if "index_size_mb" in comparison_df.columns:
        smallest = comparison_df.loc[comparison_df["index_size_mb"].idxmin()]
        analysis.append(f"**Smallest Index**: {smallest['Config']} ({smallest['index_size_mb']:.2f}MB)")
    
    return "\n".join(analysis)


def main():
    parser = argparse.ArgumentParser(description="Generate equal budget comparison report")
    parser.add_argument("--budget", type=float, default=500,
                       help="Target index size budget in MB")
    parser.add_argument("--budgets", type=float, nargs="+",
                       help="Multiple budget values to compare")
    parser.add_argument("--faiss-dir", default="results/faiss_benchmark",
                       help="FAISS results directory")
    parser.add_argument("--colbert-dir", default="results/colbert_benchmark",
                       help="ColBERT results directory")
    parser.add_argument("--plaid-dir", default="results/plaid_benchmark",
                       help="PLAID results directory")
    parser.add_argument("--out-dir", default="results/equal_budget_comparison",
                       help="Output directory")
    args = parser.parse_args()
    
    print("=" * 70)
    print("📊 Equal Budget Comparison Report Generator")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load all results
    print("\n📥 Loading benchmark results...")
    faiss_df = load_faiss_results(args.faiss_dir)
    colbert_df = load_colbert_results(args.colbert_dir)
    plaid_df = load_plaid_results(args.plaid_dir)
    
    # Check what we have
    available = []
    if faiss_df is not None and len(faiss_df) > 0:
        available.append(f"FAISS ({len(faiss_df)} rows)")
    if colbert_df is not None and len(colbert_df) > 0:
        available.append(f"ColBERT ({len(colbert_df)} rows)")
    if plaid_df is not None and len(plaid_df) > 0:
        available.append(f"PLAID ({len(plaid_df)} rows)")
    
    if not available:
        print("\n❌ No benchmark results found. Run the benchmarks first:")
        print("   python main_faiss_benchmark.py --config configs/faiss_benchmark/full_grid.yml")
        print("   python main_colbert_benchmark.py --config configs/colbert_benchmark/full_grid.yml")
        print("   python main_plaid_benchmark.py --config configs/plaid_benchmark/full_grid.yml")
        return
    
    print(f"  ✓ Found: {', '.join(available)}")
    
    # Determine budgets
    budgets = args.budgets if args.budgets else [args.budget]
    
    # Generate reports for each budget
    all_comparisons = {}
    
    for budget in budgets:
        print(f"\n📊 Generating comparison at {budget}MB budget...")
        
        comparison = generate_comparison_table(
            faiss_df, colbert_df, plaid_df, budget
        )
        
        if len(comparison) > 0:
            all_comparisons[budget] = comparison
            
            # Print table
            print(f"\n{'='*80}")
            print(f"📊 COMPARISON AT ~{budget}MB INDEX SIZE")
            print("=" * 80)
            print(comparison.to_string(index=False))
            
            # Print analysis
            print(f"\n📈 Analysis:")
            print(generate_tradeoff_analysis(comparison))
            
            # Save to CSV
            csv_path = os.path.join(args.out_dir, f"comparison_{int(budget)}mb.csv")
            comparison.to_csv(csv_path, index=False)
            print(f"\n✅ Saved: {csv_path}")
    
    # Generate combined report
    if all_comparisons:
        report = {
            "generated_at": datetime.now().isoformat(),
            "budgets_mb": budgets,
            "sources": {
                "faiss": args.faiss_dir,
                "colbert": args.colbert_dir,
                "plaid": args.plaid_dir
            },
            "comparisons": {
                f"{int(b)}mb": df.to_dict(orient="records")
                for b, df in all_comparisons.items()
            }
        }
        
        report_path = os.path.join(args.out_dir, "equal_budget_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Saved combined report: {report_path}")
    
    print("\n✅ Equal budget comparison complete!")


if __name__ == "__main__":
    main()
