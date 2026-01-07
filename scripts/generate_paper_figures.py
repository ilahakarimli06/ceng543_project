#!/usr/bin/env python3
"""
Paper Figure Generation Script
Generates publication-ready visualizations from experiment results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palettes
PALETTE = {
    'LED': '#2E86AB',
    'BigBird': '#A23B72',
    'LongT5': '#F18F01',
    'flat': '#1a9850',
    'ivf': '#91cf60',
    'ivf_pq': '#d9ef8b',
    'ColBERT': '#fc8d59',
    'PLAID': '#d73027',
    'BGE': '#4575b4',
    'MiniLM': '#fdae61',
}

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
OUTPUT_DIR = RESULTS_DIR / 'paper_figures'


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    for dataset in ['arxiv', 'longform']:
        (OUTPUT_DIR / dataset).mkdir(parents=True, exist_ok=True)


def load_sliding_window_results(dataset: str, length_category: str = 'extra_long') -> pd.DataFrame:
    """Load and aggregate sliding window results for a dataset and length category."""
    results_path = RESULTS_DIR / 'grid' / f'{dataset}_{length_category}'
    if not results_path.exists():
        print(f"Warning: Path {results_path} does not exist")
        return pd.DataFrame()
    
    all_data = []
    for csv_file in results_path.glob('*.csv'):
        if '_samples' in str(csv_file):
            continue
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def load_chunk_retrieval_results(dataset: str) -> pd.DataFrame:
    """Load chunk retrieval results for a dataset."""
    results_path = RESULTS_DIR / 'chunk_retrieval' / f'{dataset}_extra_long'
    if not results_path.exists():
        print(f"Warning: Path {results_path} does not exist")
        return pd.DataFrame()
    
    all_data = []
    for csv_file in results_path.glob('*.csv'):
        if '_samples' in str(csv_file):
            continue
        try:
            df = pd.read_csv(csv_file)
            # Extract model info from filename
            fname = csv_file.stem
            if 'bigbird' in fname.lower():
                df['sum_model'] = 'BigBird'
            elif 'longt5' in fname.lower():
                df['sum_model'] = 'LongT5'
            else:
                df['sum_model'] = 'LED'
            
            if 'bge' in fname.lower():
                df['embed_model'] = 'BGE'
            elif 'minilm' in fname.lower():
                df['embed_model'] = 'MiniLM'
            else:
                df['embed_model'] = 'Unknown'
            
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def load_faiss_results(dataset: str) -> pd.DataFrame:
    """Load FAISS benchmark results."""
    csv_path = RESULTS_DIR / 'faiss_benchmark_v2' / dataset / 'benchmark_aggregated.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def load_colbert_results(dataset: str) -> pd.DataFrame:
    """Load ColBERT benchmark results."""
    csv_path = RESULTS_DIR / 'colbert_benchmark' / dataset / 'benchmark_full.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def load_plaid_results(dataset: str) -> pd.DataFrame:
    """Load PLAID benchmark results."""
    csv_path = RESULTS_DIR / 'plaid_benchmark' / dataset / 'benchmark_aggregated.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


# ============================================================================
# GRAPH 1: Sliding Window Heatmap
# ============================================================================

def plot_sliding_window_heatmap(dataset: str):
    """Create heatmap of ROUGE-L scores for window size × overlap configurations."""
    df = load_sliding_window_results(dataset)
    if df.empty:
        print(f"No sliding window data for {dataset}")
        return
    
    # Separate LED and BigBird results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Find global max for consistent color scale
    global_max = df['rougeL'].max() if 'rougeL' in df.columns else 0.1
    
    for idx, model_key in enumerate(['LED', 'BigBird']):
        ax = axes[idx]
        
        # Filter by model using contains for flexibility
        if 'model_family' in df.columns:
            model_df = df[df['model_family'].str.contains(model_key, case=False, na=False)]
        else:
            model_df = df[df['model'].str.contains(model_key, case=False, na=False)]
        
        if model_df.empty:
            ax.text(0.5, 0.5, f'No {model_key.upper()} data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{model_key.upper()} - {dataset.upper()}')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Aggregate by window and overlap
        if 'window' in model_df.columns and 'overlap' in model_df.columns:
            pivot = model_df.groupby(['window', 'overlap'])['rougeL'].mean().unstack()
            # Sort index and columns for better visualization
            pivot = pivot.sort_index(ascending=True)
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        else:
            pivot = pd.DataFrame()
        
        if pivot.empty:
            ax.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{model_key.upper()}')
            continue
        
        # Auto-scale color range based on actual data + add some padding
        vmin = pivot.min().min() * 0.9
        vmax = pivot.max().max() * 1.1
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, 
                    cbar_kws={'label': 'ROUGE-L', 'shrink': 0.8}, 
                    vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor='white',
                    annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        ax.set_title(f'{model_key} - {dataset.upper()}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Overlap', fontsize=11)
        ax.set_ylabel('Window Size', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / dataset / 'sliding_window_heatmap.pdf')
    plt.savefig(OUTPUT_DIR / dataset / 'sliding_window_heatmap.png')
    plt.close()
    print(f"  Saved: sliding_window_heatmap.pdf")


# ============================================================================
# GRAPH 2: Quality-Latency Scatter Plot
# ============================================================================

def plot_quality_latency_scatter(dataset: str):
    """Create scatter plot of ROUGE-L vs latency with LED vs BigBird comparison."""
    df = load_sliding_window_results(dataset)
    if df.empty:
        print(f"No sliding window data for {dataset}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define markers for models and sizes for window
    model_markers = {'LED': 'o', 'BigBird': 's'}
    colors_map = {'LED': PALETTE['LED'], 'BigBird': PALETTE['BigBird']}
    window_sizes = {512: 80, 1024: 140, 2048: 220}
    
    legend_elements = []
    
    for model_key in ['LED', 'BigBird']:
        # Filter by model
        if 'model_family' in df.columns:
            model_df = df[df['model_family'].str.contains(model_key, case=False, na=False)]
        else:
            model_df = df[df['model'].str.contains(model_key, case=False, na=False)]
        
        if model_df.empty:
            continue
        
        # Group by configuration (only window and overlap, ignore global_tokens)
        grouped = model_df.groupby(['window', 'overlap']).agg({
            'rougeL': 'mean',
            'latency': 'mean',
        }).reset_index()
        
        # Plot points with size based on window
        for _, row in grouped.iterrows():
            w = int(row['window'])
            size = window_sizes.get(w, 100)
            ax.scatter(row['latency'], row['rougeL'],
                      c=colors_map[model_key], marker=model_markers[model_key],
                      s=size, alpha=0.8, edgecolors='black', linewidths=1.5)
        
        # Add to legend
        legend_elements.append(plt.scatter([], [], c=colors_map[model_key], 
                                           marker=model_markers[model_key], s=140, 
                                           edgecolors='black', label=model_key))
    
    # Add window size legend
    for w, size in window_sizes.items():
        legend_elements.append(plt.scatter([], [], c='gray', marker='o', s=size, 
                                           alpha=0.5, edgecolors='black', label=f'W={w}'))
    
    ax.set_xlabel('Mean Latency (seconds)', fontsize=13)
    ax.set_ylabel('Mean ROUGE-L', fontsize=13)
    ax.set_title(f'Quality-Latency Trade-off: LED vs BigBird - {dataset.upper()}', fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    model_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['LED'], 
                           markersize=12, markeredgecolor='black', label='LED'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor=PALETTE['BigBird'], 
                           markersize=12, markeredgecolor='black', label='BigBird')]
    size_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                          markersize=s/15, markeredgecolor='black', alpha=0.5, label=f'W={w}')
                  for w, s in window_sizes.items()]
    
    legend1 = ax.legend(handles=model_legend, title='Model', loc='upper right', fontsize=10)
    ax.add_artist(legend1)
    ax.legend(handles=size_legend, title='Window Size', loc='lower left', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / dataset / 'quality_latency_scatter.pdf')
    plt.savefig(OUTPUT_DIR / dataset / 'quality_latency_scatter.png')
    plt.close()
    print(f"  Saved: quality_latency_scatter.pdf")


# ============================================================================
# GRAPH 3: Chunk Retrieval Model Comparison
# ============================================================================

def plot_chunk_retrieval_comparison(dataset: str):
    """Create bar chart comparing chunk retrieval models."""
    df = load_chunk_retrieval_results(dataset)
    if df.empty:
        print(f"No chunk retrieval data for {dataset}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Aggregate by model combination
    grouped = df.groupby(['sum_model', 'embed_model']).agg({
        'rougeL': ['mean', 'std'],
        'bertscore_f1': ['mean', 'std'],
    }).reset_index()
    grouped.columns = ['sum_model', 'embed_model', 'rouge_mean', 'rouge_std', 'bert_mean', 'bert_std']
    
    # Create combined label
    grouped['label'] = grouped['sum_model'] + '+' + grouped['embed_model']
    
    x = np.arange(len(grouped))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, grouped['rouge_mean'], width, 
                   yerr=grouped['rouge_std'], label='ROUGE-L', 
                   color=PALETTE['LED'], capsize=3)
    bars2 = ax.bar(x + width/2, grouped['bert_mean'], width, 
                   yerr=grouped['bert_std'], label='BERTScore F1', 
                   color=PALETTE['BigBird'], capsize=3)
    
    ax.set_ylabel('Score')
    ax.set_xlabel('Model Configuration')
    ax.set_title(f'Chunk Retrieval Model Comparison - {dataset.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['label'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / dataset / 'chunk_retrieval_comparison.pdf')
    plt.savefig(OUTPUT_DIR / dataset / 'chunk_retrieval_comparison.png')
    plt.close()
    print(f"  Saved: chunk_retrieval_comparison.pdf")


# ============================================================================
# GRAPH 4: Retrieval Methods Comparison (FAISS vs ColBERT vs PLAID)
# ============================================================================

def plot_retrieval_methods_comparison(dataset: str):
    """Create scatter plot of index size vs downstream quality for retrieval methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Load FAISS results
    faiss_df = load_faiss_results(dataset)
    if not faiss_df.empty:
        for idx_type in faiss_df['index_type'].unique():
            mask = faiss_df['index_type'] == idx_type
            ax.scatter(faiss_df.loc[mask, 'index_size_mb'], 
                      faiss_df.loc[mask, 'downstream_rouge_l'],
                      label=f'FAISS-{idx_type}', s=60, alpha=0.7,
                      color=PALETTE.get(idx_type, 'gray'))
    
    # Load ColBERT results
    colbert_df = load_colbert_results(dataset)
    if not colbert_df.empty:
        ax.scatter(colbert_df['index_size_mb'].mean(), 
                  colbert_df['downstream_rouge_l'].mean(),
                  label='ColBERT', s=150, marker='s', color=PALETTE['ColBERT'],
                  edgecolors='black', linewidths=2)
    
    # Load PLAID results
    plaid_df = load_plaid_results(dataset)
    if not plaid_df.empty:
        ax.scatter(plaid_df['plaid_index_size_mb'], 
                  plaid_df['downstream_rouge_l'],
                  label='PLAID', s=100, marker='^', color=PALETTE['PLAID'],
                  edgecolors='black', linewidths=1.5)
    
    ax.set_xlabel('Index Size (MB)')
    ax.set_ylabel('Downstream ROUGE-L')
    ax.set_title(f'Retrieval Methods: Index Size vs Quality - {dataset.upper()}')
    ax.set_xscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / dataset / 'retrieval_methods_comparison.pdf')
    plt.savefig(OUTPUT_DIR / dataset / 'retrieval_methods_comparison.png')
    plt.close()
    print(f"  Saved: retrieval_methods_comparison.pdf")


# ============================================================================
# GRAPH 5: FAISS Pareto Frontier
# ============================================================================

def plot_faiss_pareto(dataset: str):
    """Create plot showing FAISS recall-latency trade-off with Pareto frontier."""
    df = load_faiss_results(dataset)
    if df.empty:
        print(f"No FAISS data for {dataset}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot by index type
    for idx_type in df['index_type'].unique():
        mask = df['index_type'] == idx_type
        subset = df[mask]
        ax.scatter(subset['total_retrieval_ms'], 
                  subset['recall_at_10'],
                  label=idx_type.upper(), s=80, alpha=0.7,
                  color=PALETTE.get(idx_type, 'gray'))
    
    # Compute and plot Pareto frontier
    points = df[['total_retrieval_ms', 'recall_at_10']].values
    pareto_mask = np.ones(len(points), dtype=bool)
    
    for i, (latency_i, recall_i) in enumerate(points):
        for j, (latency_j, recall_j) in enumerate(points):
            if i != j:
                # Point j dominates i if j has lower latency AND higher recall
                if latency_j <= latency_i and recall_j >= recall_i:
                    if latency_j < latency_i or recall_j > recall_i:
                        pareto_mask[i] = False
                        break
    
    pareto_df = df[pareto_mask].sort_values('total_retrieval_ms')
    ax.plot(pareto_df['total_retrieval_ms'], pareto_df['recall_at_10'], 
            'k--', alpha=0.5, linewidth=2, label='Pareto Frontier')
    
    ax.set_xlabel('Retrieval Latency (ms)')
    ax.set_ylabel('Recall@10')
    ax.set_title(f'FAISS Recall-Latency Trade-off - {dataset.upper()}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / dataset / 'faiss_pareto.pdf')
    plt.savefig(OUTPUT_DIR / dataset / 'faiss_pareto.png')
    plt.close()
    print(f"  Saved: faiss_pareto.pdf")


# ============================================================================
# GRAPH 6: GPU Memory Comparison
# ============================================================================

def plot_gpu_memory_comparison(dataset: str):
    """Create bar chart comparing GPU memory usage across methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    gpu_memory = []
    colors = []
    
    # Sliding window (LED)
    sw_df = load_sliding_window_results(dataset)
    if not sw_df.empty and 'gpu_peak_gb' in sw_df.columns:
        if 'model_family' in sw_df.columns:
            led_df = sw_df[sw_df['model_family'].str.lower() == 'led']
            bigbird_df = sw_df[sw_df['model_family'].str.lower() == 'bigbird']
        else:
            led_df = sw_df[sw_df['model'].str.contains('led', case=False, na=False)]
            bigbird_df = sw_df[sw_df['model'].str.contains('bigbird', case=False, na=False)]
        
        if not led_df.empty:
            methods.append('LED (Sliding)')
            gpu_memory.append(led_df['gpu_peak_gb'].mean())
            colors.append(PALETTE['LED'])
        
        if not bigbird_df.empty:
            methods.append('BigBird (Sliding)')
            gpu_memory.append(bigbird_df['gpu_peak_gb'].mean())
            colors.append(PALETTE['BigBird'])
    
    # Chunk retrieval
    cr_df = load_chunk_retrieval_results(dataset)
    if not cr_df.empty and 'gpu_peak_gb' in cr_df.columns:
        for model in cr_df['sum_model'].unique():
            model_df = cr_df[cr_df['sum_model'] == model]
            methods.append(f'{model} (Chunk)')
            gpu_memory.append(model_df['gpu_peak_gb'].mean())
            colors.append(PALETTE.get(model, 'gray'))
    
    # ColBERT
    colbert_df = load_colbert_results(dataset)
    if not colbert_df.empty and 'gpu_peak_gb' in colbert_df.columns:
        methods.append('ColBERT')
        gpu_memory.append(colbert_df['gpu_peak_gb'].mean())
        colors.append(PALETTE['ColBERT'])
    
    if not methods:
        print(f"No GPU memory data for {dataset}")
        return
    
    x = np.arange(len(methods))
    bars = ax.bar(x, gpu_memory, color=colors, edgecolor='black')
    
    ax.set_ylabel('Peak GPU Memory (GB)')
    ax.set_xlabel('Method')
    ax.set_title(f'GPU Memory Usage Comparison - {dataset.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, gpu_memory):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / dataset / 'gpu_memory_comparison.pdf')
    plt.savefig(OUTPUT_DIR / dataset / 'gpu_memory_comparison.png')
    plt.close()
    print(f"  Saved: gpu_memory_comparison.pdf")


# ============================================================================
# GRAPH 7: Embedding Model Comparison
# ============================================================================

def plot_embedding_comparison(dataset: str):
    """Compare BGE vs MiniLM embedding models."""
    df = load_chunk_retrieval_results(dataset)
    if df.empty:
        print(f"No chunk retrieval data for {dataset}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Aggregate by summarization model and embedding model
    grouped = df.groupby(['sum_model', 'embed_model'])['rougeL'].agg(['mean', 'std']).reset_index()
    
    models = grouped['sum_model'].unique()
    x = np.arange(len(models))
    width = 0.35
    
    bge_data = grouped[grouped['embed_model'] == 'BGE'].set_index('sum_model')
    minilm_data = grouped[grouped['embed_model'] == 'MiniLM'].set_index('sum_model')
    
    bge_means = [bge_data.loc[m, 'mean'] if m in bge_data.index else 0 for m in models]
    bge_stds = [bge_data.loc[m, 'std'] if m in bge_data.index else 0 for m in models]
    minilm_means = [minilm_data.loc[m, 'mean'] if m in minilm_data.index else 0 for m in models]
    minilm_stds = [minilm_data.loc[m, 'std'] if m in minilm_data.index else 0 for m in models]
    
    ax.bar(x - width/2, bge_means, width, yerr=bge_stds, 
           label='BGE', color=PALETTE['BGE'], capsize=3)
    ax.bar(x + width/2, minilm_means, width, yerr=minilm_stds, 
           label='MiniLM', color=PALETTE['MiniLM'], capsize=3)
    
    ax.set_ylabel('ROUGE-L')
    ax.set_xlabel('Summarization Model')
    ax.set_title(f'Embedding Model Comparison - {dataset.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / dataset / 'embedding_comparison.pdf')
    plt.savefig(OUTPUT_DIR / dataset / 'embedding_comparison.png')
    plt.close()
    print(f"  Saved: embedding_comparison.pdf")


# ============================================================================
# GRAPH 8: Combined Overview (All Methods on Same Plot)
# ============================================================================

def plot_combined_overview(dataset: str):
    """Create a combined overview plot comparing all methods."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_methods = []
    all_rouge = []
    all_colors = []
    all_categories = []
    
    # Sliding Window
    sw_df = load_sliding_window_results(dataset)
    if not sw_df.empty:
        if 'model_family' in sw_df.columns:
            for model in sw_df['model_family'].unique():
                model_df = sw_df[sw_df['model_family'] == model]
                all_methods.append(f'{model}\n(Sliding)')
                all_rouge.append(model_df['rougeL'].mean())
                all_colors.append(PALETTE.get(model.upper(), PALETTE.get(model, 'gray')))
                all_categories.append('Sliding Window')
    
    # Chunk Retrieval
    cr_df = load_chunk_retrieval_results(dataset)
    if not cr_df.empty:
        for model in cr_df['sum_model'].unique():
            model_df = cr_df[cr_df['sum_model'] == model]
            all_methods.append(f'{model}\n(Chunk)')
            all_rouge.append(model_df['rougeL'].mean())
            all_colors.append(PALETTE.get(model, 'gray'))
            all_categories.append('Chunk Retrieval')
    
    # FAISS (best config)
    faiss_df = load_faiss_results(dataset)
    if not faiss_df.empty:
        best_faiss = faiss_df.loc[faiss_df['downstream_rouge_l'].idxmax()]
        all_methods.append(f"FAISS\n({best_faiss['index_type']})")
        all_rouge.append(best_faiss['downstream_rouge_l'])
        all_colors.append(PALETTE.get(best_faiss['index_type'], 'gray'))
        all_categories.append('Retrieval')
    
    # ColBERT
    colbert_df = load_colbert_results(dataset)
    if not colbert_df.empty:
        all_methods.append('ColBERT')
        all_rouge.append(colbert_df['downstream_rouge_l'].mean())
        all_colors.append(PALETTE['ColBERT'])
        all_categories.append('Retrieval')
    
    # PLAID (best config)
    plaid_df = load_plaid_results(dataset)
    if not plaid_df.empty:
        best_plaid = plaid_df.loc[plaid_df['downstream_rouge_l'].idxmax()]
        all_methods.append('PLAID')
        all_rouge.append(best_plaid['downstream_rouge_l'])
        all_colors.append(PALETTE['PLAID'])
        all_categories.append('Retrieval')
    
    if not all_methods:
        print(f"No data for combined overview for {dataset}")
        return
    
    x = np.arange(len(all_methods))
    bars = ax.bar(x, all_rouge, color=all_colors, edgecolor='black')
    
    ax.set_ylabel('ROUGE-L')
    ax.set_xlabel('Method')
    ax.set_title(f'Method Comparison Overview - {dataset.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(all_methods, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, all_rouge):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / dataset / 'combined_overview.pdf')
    plt.savefig(OUTPUT_DIR / dataset / 'combined_overview.png')
    plt.close()
    print(f"  Saved: combined_overview.pdf")


# ============================================================================
# Main Entry Point
# ============================================================================

def generate_all_figures():
    """Generate all paper figures for both datasets."""
    ensure_output_dirs()
    
    for dataset in ['arxiv', 'longform']:
        print(f"\n{'='*60}")
        print(f"Generating figures for: {dataset.upper()}")
        print('='*60)
        
        plot_sliding_window_heatmap(dataset)
        plot_quality_latency_scatter(dataset)
        plot_chunk_retrieval_comparison(dataset)
        plot_retrieval_methods_comparison(dataset)
        plot_faiss_pareto(dataset)
        plot_gpu_memory_comparison(dataset)
        plot_embedding_comparison(dataset)
        plot_combined_overview(dataset)
    
    print(f"\n{'='*60}")
    print(f"All figures saved to: {OUTPUT_DIR}")
    print('='*60)


if __name__ == '__main__':
    generate_all_figures()
