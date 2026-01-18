#!/bin/bash
# =============================================================================
# Paper Tables Reproduction Script
# =============================================================================
# This script reproduces all benchmark results from the paper.
# Usage: bash scripts/reproduce_tables.sh
#
# Prerequisites:
#   1. Install dependencies: uv pip install -r requirements.txt
#   2. Prepare datasets: uv run python scripts/dataset_prep.py --ds both --n 800
#   3. Filter and clean: See README.md Dataset Preparation section
# =============================================================================

set -e  # Exit on error

echo "============================================"
echo "📊 Paper Tables Reproduction Script"
echo "============================================"
echo ""

# Check if data exists
if [ ! -d "src/data/cleaned" ]; then
    echo "❌ Error: Dataset not found. Please run dataset preparation first."
    echo "   See README.md for instructions."
    exit 1
fi

# =============================================================================
# Table 1-2: Grid Search (Sliding Window Summarization)
# =============================================================================
echo ""
echo "📊 [1/4] Running Grid Search experiments (Tables 1-2)..."
echo "============================================"

for cfg in configs/sliding/grid/*.yml; do
    echo "  → Running: $(basename $cfg)"
    uv run python main.py --config "$cfg"
done

echo "  ✓ Grid Search complete. Results: results/grid/"

# Generate analysis
echo "  → Analyzing grid results..."
uv run python scripts/analyze_grid_results.py

# =============================================================================
# Table 3-4: Chunk Retrieval Experiments
# =============================================================================
echo ""
echo "📊 [2/4] Running Chunk Retrieval experiments (Tables 3-4)..."
echo "============================================"

for cfg in configs/chunk_retrieval/*.yml; do
    echo "  → Running: $(basename $cfg)"
    uv run python main_chunk_retrieval.py --config "$cfg"
done

echo "  ✓ Chunk Retrieval complete. Results: results/chunk_retrieval/"

# =============================================================================
# Table 5: FAISS Benchmark (Dense Retrieval)
# =============================================================================
echo ""
echo "📊 [3/4] Running FAISS Benchmark (Table 5)..."
echo "============================================"

# ArXiv dataset
echo "  → ArXiv extra-long..."
uv run python main_faiss_benchmark_v2.py --config configs/faiss_benchmark/full_grid_v2.yml

# Longform dataset
echo "  → Longform extra-long..."
uv run python main_faiss_benchmark_v2.py --config configs/faiss_benchmark/full_grid_v2_longform.yml

echo "  ✓ FAISS Benchmark complete. Results: results/faiss_benchmark_v2/"

# =============================================================================
# Table 6: ColBERT/PLAID Benchmark
# =============================================================================
echo ""
echo "📊 [4/4] Running ColBERT/PLAID Benchmark (Table 6)..."
echo "============================================"

# ColBERT
echo "  → ColBERT benchmark..."
uv run python main_colbert_benchmark.py --config configs/colbert_benchmark/full_grid.yml
uv run python main_colbert_benchmark.py --config configs/colbert_benchmark/full_grid_longform.yml

# PLAID
echo "  → PLAID benchmark..."
uv run python main_plaid_benchmark.py --config configs/plaid_benchmark/full_grid.yml
uv run python main_plaid_benchmark.py --config configs/plaid_benchmark/full_grid_longform.yml

echo "  ✓ ColBERT/PLAID complete. Results: results/colbert_benchmark/, results/plaid_benchmark/"

# =============================================================================
# Generate Equal Budget Comparison Report
# =============================================================================
echo ""
echo "📊 Generating Equal Budget Comparison Report..."
echo "============================================"

uv run python generate_equal_budget_report.py --budgets 100 500 1000

echo "  ✓ Report complete. Results: results/equal_budget_comparison/"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================"
echo "✅ ALL EXPERIMENTS COMPLETE!"
echo "============================================"
echo ""
echo "Results locations:"
echo "  • Grid Search:       results/grid/"
echo "  • Chunk Retrieval:   results/chunk_retrieval/"
echo "  • FAISS Benchmark:   results/faiss_benchmark_v2/"
echo "  • ColBERT Benchmark: results/colbert_benchmark/"
echo "  • PLAID Benchmark:   results/plaid_benchmark/"
echo "  • Comparison Report: results/equal_budget_comparison/"
echo ""
echo "To analyze results: uv run python scripts/analyze_grid_results.py"
