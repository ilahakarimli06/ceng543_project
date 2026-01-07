#!/usr/bin/env python3
"""Generate publication-ready FAISS benchmark visualizations."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif', 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.linewidth': 1.2
})

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
OUTPUT_DIR = RESULTS_DIR / 'paper_figures'

COLORS = {'FLAT': '#2ecc71', 'IVF': '#3498db', 'IVF-PQ': '#e74c3c'}

def load_faiss_data(dataset):
    path = RESULTS_DIR / 'faiss_benchmark_v2' / dataset / 'benchmark_aggregated.csv'
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def plot_faiss_bar_comparison(dataset):
    """Grouped bar chart comparing index types."""
    df = load_faiss_data(dataset)
    if df.empty:
        return
    
    # Aggregate by index type (best config per type)
    summary = []
    for idx_type in ['flat', 'ivf', 'ivf_pq']:
        subset = df[df['index_type'] == idx_type]
        best = subset.loc[subset['downstream_rouge_l'].idxmax()]
        summary.append({
            'Index': idx_type.upper().replace('_', '-'),
            'ROUGE-L': best['downstream_rouge_l'],
            'Index Size (MB)': best['index_size_mb'],
            'Latency (ms)': best['total_retrieval_ms'],
            'Recall@10': best['recall_at_10']
        })
    summary_df = pd.DataFrame(summary)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    x = np.arange(len(summary_df))
    width = 0.6
    colors = [COLORS[idx] for idx in summary_df['Index']]
    
    # ROUGE-L
    bars1 = axes[0].bar(x, summary_df['ROUGE-L'], width, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('ROUGE-L', fontsize=12)
    axes[0].set_title('Downstream Quality', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary_df['Index'], fontsize=11)
    axes[0].set_ylim(0.18, 0.25)
    for bar, val in zip(bars1, summary_df['ROUGE-L']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Index Size
    bars2 = axes[1].bar(x, summary_df['Index Size (MB)'], width, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Index Size (MB)', fontsize=12)
    axes[1].set_title('Storage Efficiency', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary_df['Index'], fontsize=11)
    axes[1].set_yscale('log')
    for bar, val in zip(bars2, summary_df['Index Size (MB)']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Latency
    bars3 = axes[2].bar(x, summary_df['Latency (ms)'], width, color=colors, edgecolor='black', linewidth=1.2)
    axes[2].set_ylabel('Latency (ms)', fontsize=12)
    axes[2].set_title('Retrieval Speed', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(summary_df['Index'], fontsize=11)
    for bar, val in zip(bars3, summary_df['Latency (ms)']):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for ax in axes:
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle(f'FAISS Index Comparison - {dataset.upper()}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    (OUTPUT_DIR / dataset).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / dataset / 'faiss_index_comparison.pdf')
    plt.savefig(OUTPUT_DIR / dataset / 'faiss_index_comparison.png')
    plt.close()
    print(f"  Saved: faiss_index_comparison.pdf for {dataset}")

def plot_compression_line(dataset):
    """Line plot showing compression ratio vs quality."""
    df = load_faiss_data(dataset)
    if df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Get flat baseline
    flat_size = df[df['index_type'] == 'flat']['index_size_mb'].mean()
    df['compression_ratio'] = flat_size / df['index_size_mb']
    
    # Group by index type and plot lines
    for idx_type, marker, color in [('ivf', 's', '#3498db'), ('ivf_pq', '^', '#e74c3c')]:
        subset = df[df['index_type'] == idx_type].sort_values('compression_ratio')
        ax.plot(subset['compression_ratio'], subset['downstream_rouge_l'], 
               marker=marker, color=color, linewidth=2, markersize=8, 
               label=idx_type.upper().replace('_', '-'), alpha=0.8)
    
    # Flat baseline
    flat_rouge = df[df['index_type'] == 'flat']['downstream_rouge_l'].mean()
    ax.axhline(y=flat_rouge, color='#2ecc71', linestyle='--', linewidth=2, label='Flat (baseline)')
    
    ax.set_xlabel('Compression Ratio (×)', fontsize=12)
    ax.set_ylabel('ROUGE-L', fontsize=12)
    ax.set_title(f'Compression vs Quality Trade-off - {dataset.upper()}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / dataset / 'faiss_compression_line.pdf')
    plt.savefig(OUTPUT_DIR / dataset / 'faiss_compression_line.png')
    plt.close()
    print(f"  Saved: faiss_compression_line.pdf for {dataset}")

if __name__ == '__main__':
    for dataset in ['arxiv', 'longform']:
        plot_faiss_bar_comparison(dataset)
        plot_compression_line(dataset)
    print("Done!")
