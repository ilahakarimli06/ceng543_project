#!/usr/bin/env python3
"""Generate PLAID benchmark visualization: Compression vs Quality line plot."""

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

def plot_plaid_comparison():
    """Line plot comparing PLAID compression levels vs quality."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'arxiv': '#e74c3c', 'longform': '#3498db'}
    
    for dataset, color in colors.items():
        path = RESULTS_DIR / 'plaid_benchmark' / dataset / 'benchmark_aggregated.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df.sort_values('compression_ratio')
        
        # Left: Compression vs ROUGE-L
        axes[0].plot(df['compression_ratio'], df['downstream_rouge_l'], 
                    marker='o', linewidth=2.5, markersize=10, color=color,
                    label=dataset.upper(), alpha=0.9)
        
        # Right: Compression vs Index Size
        axes[1].plot(df['compression_ratio'], df['plaid_index_size_mb'],
                    marker='s', linewidth=2.5, markersize=10, color=color,
                    label=dataset.upper(), alpha=0.9)
    
    # Left plot styling
    axes[0].set_xlabel('Compression Ratio (×)', fontsize=12)
    axes[0].set_ylabel('ROUGE-L', fontsize=12)
    axes[0].set_title('Quality vs Compression', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # ColBERT baseline
    axes[0].axhline(y=0.237, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].annotate('ColBERT (88.5MB)', xy=(150, 0.237), fontsize=9, color='gray')
    
    # Right plot styling
    axes[1].set_xlabel('Compression Ratio (×)', fontsize=12)
    axes[1].set_ylabel('Index Size (MB)', fontsize=12)
    axes[1].set_title('Storage Efficiency', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # ColBERT baseline
    axes[1].axhline(y=88.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].annotate('ColBERT baseline', xy=(150, 88.5), fontsize=9, color='gray')
    
    fig.suptitle('PLAID: Token Pruning + PQ Compression', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    for dataset in ['arxiv', 'longform']:
        (OUTPUT_DIR / dataset).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'arxiv' / 'plaid_compression_analysis.pdf')
    plt.savefig(OUTPUT_DIR / 'arxiv' / 'plaid_compression_analysis.png')
    plt.close()
    print("  Saved: plaid_compression_analysis.pdf")

if __name__ == '__main__':
    plot_plaid_comparison()
    print("Done!")
