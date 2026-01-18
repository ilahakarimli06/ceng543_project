# =============================================================================
# Paper Tables Reproduction Script (Windows PowerShell)
# =============================================================================
# This script reproduces all benchmark results from the paper.
# Usage: .\scripts\reproduce_tables.ps1
#
# Prerequisites:
#   1. Install dependencies: uv pip install -r requirements.txt
#   2. Prepare datasets: uv run python scripts/dataset_prep.py --ds both --n 800
#   3. Filter and clean: See README.md Dataset Preparation section
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "📊 Paper Tables Reproduction Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if data exists
if (-not (Test-Path "src/data/cleaned")) {
    Write-Host "❌ Error: Dataset not found. Please run dataset preparation first." -ForegroundColor Red
    Write-Host "   See README.md for instructions."
    exit 1
}

# =============================================================================
# Table 1-2: Grid Search (Sliding Window Summarization)
# =============================================================================
Write-Host ""
Write-Host "📊 [1/4] Running Grid Search experiments (Tables 1-2)..." -ForegroundColor Yellow
Write-Host "============================================"

$gridConfigs = Get-ChildItem -Path "configs/sliding/grid" -Filter "*.yml"
$total = $gridConfigs.Count
$current = 0

foreach ($cfg in $gridConfigs) {
    $current++
    Write-Host "  → [$current/$total] $($cfg.Name)" -ForegroundColor Gray
    python main.py --config $cfg.FullName
}

Write-Host "  ✓ Grid Search complete. Results: results/grid/" -ForegroundColor Green

# Generate analysis
Write-Host "  → Analyzing grid results..."
python scripts/analyze_grid_results.py

# =============================================================================
# Table 3-4: Chunk Retrieval Experiments
# =============================================================================
Write-Host ""
Write-Host "📊 [2/4] Running Chunk Retrieval experiments (Tables 3-4)..." -ForegroundColor Yellow
Write-Host "============================================"

$chunkConfigs = Get-ChildItem -Path "configs/chunk_retrieval" -Filter "*.yml"
foreach ($cfg in $chunkConfigs) {
    Write-Host "  → $($cfg.Name)" -ForegroundColor Gray
    python main_chunk_retrieval.py --config $cfg.FullName
}

Write-Host "  ✓ Chunk Retrieval complete. Results: results/chunk_retrieval/" -ForegroundColor Green

# =============================================================================
# Table 5: FAISS Benchmark (Dense Retrieval)
# =============================================================================
Write-Host ""
Write-Host "📊 [3/4] Running FAISS Benchmark (Table 5)..." -ForegroundColor Yellow
Write-Host "============================================"

# ArXiv dataset
Write-Host "  → ArXiv extra-long..." -ForegroundColor Gray
python main_faiss_benchmark_v2.py --config configs/faiss_benchmark/full_grid_v2.yml

# Longform dataset
Write-Host "  → Longform extra-long..." -ForegroundColor Gray
python main_faiss_benchmark_v2.py --config configs/faiss_benchmark/full_grid_v2_longform.yml

Write-Host "  ✓ FAISS Benchmark complete. Results: results/faiss_benchmark_v2/" -ForegroundColor Green

# =============================================================================
# Table 6: ColBERT/PLAID Benchmark
# =============================================================================
Write-Host ""
Write-Host "📊 [4/4] Running ColBERT/PLAID Benchmark (Table 6)..." -ForegroundColor Yellow
Write-Host "============================================"

# ColBERT
Write-Host "  → ColBERT benchmark..." -ForegroundColor Gray
python main_colbert_benchmark.py --config configs/colbert_benchmark/full_grid.yml
python main_colbert_benchmark.py --config configs/colbert_benchmark/full_grid_longform.yml

# PLAID
Write-Host "  → PLAID benchmark..." -ForegroundColor Gray
python main_plaid_benchmark.py --config configs/plaid_benchmark/full_grid.yml
python main_plaid_benchmark.py --config configs/plaid_benchmark/full_grid_longform.yml

Write-Host "  ✓ ColBERT/PLAID complete." -ForegroundColor Green

# =============================================================================
# Generate Equal Budget Comparison Report
# =============================================================================
Write-Host ""
Write-Host "📊 Generating Equal Budget Comparison Report..." -ForegroundColor Yellow
Write-Host "============================================"

python generate_equal_budget_report.py --budgets 100 500 1000

Write-Host "  ✓ Report complete." -ForegroundColor Green

# =============================================================================
# Summary
# =============================================================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "✅ ALL EXPERIMENTS COMPLETE!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results locations:"
Write-Host "  • Grid Search:       results/grid/"
Write-Host "  • Chunk Retrieval:   results/chunk_retrieval/"
Write-Host "  • FAISS Benchmark:   results/faiss_benchmark_v2/"
Write-Host "  • ColBERT Benchmark: results/colbert_benchmark/"
Write-Host "  • PLAID Benchmark:   results/plaid_benchmark/"
Write-Host "  • Comparison Report: results/equal_budget_comparison/"
Write-Host ""
Write-Host "To analyze results: python scripts/analyze_grid_results.py"
