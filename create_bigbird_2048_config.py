"""
Test BigBird-Pegasus with window_size=2048
Larger windows for better context
"""
import yaml
import sys

# Load base config
config_path = "configs/sliding/grid/arxiv_extra_long_w1024_ov256_g16.yml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# Override window size to 2048
cfg["window_size"] = 2048
cfg["overlap"] = 512  # 25% overlap
cfg["out_csv"] = "results/grid/arxiv_extra_long/bigbird_w2048_ov512.csv"

# Save modified config
output_config = "configs/sliding/grid/arxiv_extra_long_w2048_ov512_bigbird.yml"
with open(output_config, 'w') as f:
    yaml.dump(cfg, f)

print(f"✅ Created config: {output_config}")
print(f"   Window size: 2048")
print(f"   Overlap: 512 (25%)")
print(f"\nRun with:")
print(f"python test_bigbird.py --config {output_config}")
