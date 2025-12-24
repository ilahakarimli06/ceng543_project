#!/bin/bash
# Run hierarchical aggregation tests and compare with existing concat results
# Usage: bash scripts/run_aggregation_test.sh

set -e

echo "🚀 Starting Hierarchical Aggregation Test"
echo "=========================================="
echo "Note: Concat results already exist in results/grid/"
echo "      This script runs only hierarchical configs for comparison"
echo ""

# Create results directory
mkdir -p results/aggregation_test

# List of hierarchical configs to run
CONFIGS=(
    "configs/sliding/aggregation_test/led_arxiv_extra_long_hierarchical.yml"
    "configs/sliding/aggregation_test/led_longform_long_hierarchical.yml"
    "configs/sliding/aggregation_test/bigbird_arxiv_extra_long_hierarchical.yml"
    "configs/sliding/aggregation_test/bigbird_longform_long_hierarchical.yml"
    "configs/sliding/aggregation_test/led_arxiv_long_w2048_hierarchical.yml"
    "configs/sliding/aggregation_test/led_longform_extra_long_hierarchical.yml"
)

# Run each config
for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "📋 Running: $cfg"
    echo "-------------------------------------------"
    uv run python main.py --config "$cfg"
done

echo ""
echo "✅ All hierarchical tests completed!"
echo "📊 Run 'uv run python scripts/compare_aggregation.py' to analyze results"
