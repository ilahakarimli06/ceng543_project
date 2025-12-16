"""
Grid Search Config Generator
Optimized set: 42 total configs

Base configs (6 per category): All 6 categories
Extra configs (3 more): Only for extra_long categories
Note: attention_impl can be overridden via command line
"""

import os
import yaml
from pathlib import Path

# 6 categories: domain × length
CATEGORIES = [
    ("arxiv", "medium", "src/data/cleaned/cleaned_medium_examples_arxiv.jsonl"),
    ("arxiv", "long", "src/data/cleaned/cleaned_long_examples_arxiv.jsonl"),
    ("arxiv", "extra_long", "src/data/cleaned/cleaned_extra_long_examples_arxiv.jsonl"),
    ("longform", "medium", "src/data/cleaned/cleaned_medium_examples_longform.jsonl"),
    ("longform", "long", "src/data/cleaned/cleaned_long_examples_longform.jsonl"),
    ("longform", "extra_long", "src/data/cleaned/cleaned_extra_long_examples_longform.jsonl"),
]

# Base parameter combinations (for ALL categories)
BASE_COMBINATIONS = [
    (512, 0, 0),      # Small baseline
    (1024, 0, 0),     # Medium baseline
    (2048, 0, 0),     # Large baseline
    (512, 128, 16),   # Small + 25% overlap
    (1024, 256, 16),  # Medium + 25% overlap (likely optimal)
    (2048, 512, 16),  # Large + 25% overlap
]

# Extra combinations (ONLY for extra_long categories)
EXTRA_LONG_COMBINATIONS = [
    (1024, 256, 64),  # w1024 + 25% overlap + g64
    (1024, 512, 16),  # w1024 + 50% overlap + g16
    (1024, 512, 64),  # w1024 + 50% overlap + g64
]

# Base config
BASE_CONFIG = {
    "method": "sliding",
    "model_name": "allenai/led-base-16384",
    "attention_impl": "default",  # Can be overridden via --attention flag
    "task": "summ",
    "samples": 60,  # Changed from 50 to 60
    "gen_max_tokens": 256,
    "ctx_budget": 16000,
    "seed": 42,
}

def generate_configs():
    """Generate optimized set of 42 configs"""
    
    output_dir = Path("configs/sliding/grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old configs
    for old_file in output_dir.glob("*.yml"):
        old_file.unlink()
    
    count = 0
    manifest = []
    
    for domain, length, dataset_path in CATEGORIES:
        category = f"{domain}_{length}"
        is_extra_long = (length == "extra_long")
        
        # Determine which combinations to use
        if is_extra_long:
            combinations = BASE_COMBINATIONS + EXTRA_LONG_COMBINATIONS
        else:
            combinations = BASE_COMBINATIONS
        
        for window_size, overlap, global_tokens in combinations:
            count += 1
            
            # Calculate overlap ratio for display
            overlap_ratio = overlap / window_size if window_size > 0 else 0
            
            # Build config
            config = BASE_CONFIG.copy()
            config.update({
                "domain": domain,
                "length_category": length,
                "category": category,
                "dataset_path": dataset_path,
                "window_size": window_size,
                "overlap": overlap,
                "global_tokens": global_tokens,
            })
            
            # Filename
            filename = f"{category}_w{window_size}_ov{overlap}_g{global_tokens}.yml"
            filepath = output_dir / filename
            
            # CSV output path
            config["out_csv"] = f"results/grid/{category}/led_w{window_size}_ov{overlap}_g{global_tokens}.csv"
            
            # YAML header
            header = (
                f"# Optimized Grid Config {count}/42\n"
                f"# Category: {category}\n"
                f"# Domain: {domain}\n"
                f"# Length: {length}\n"
                f"# Window Size: {window_size} tokens\n"
                f"# Overlap: {overlap} tokens ({overlap_ratio*100:.0f}%)\n"
                f"# Global Tokens: {global_tokens}\n"
                f"# Note: Use --attention flag to override attention_impl\n\n"
            )
            
            # Write file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(header)
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            # Add to manifest
            manifest.append({
                "id": count,
                "filename": filename,
                "domain": domain,
                "length_category": length,
                "category": category,
                "window_size": window_size,
                "overlap": overlap,
                "overlap_ratio": f"{overlap_ratio*100:.0f}%",
                "global_tokens": global_tokens,
            })
            
            print(f"✓ [{count:2d}/42] {filename}")
    
    # Write manifest
    manifest_path = output_dir / "MANIFEST.yml"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        yaml.dump({"experiments": manifest}, f, default_flow_style=False)
    
    print(f"\n✅ {count} config dosyası oluşturuldu: {output_dir}")
    print(f"✅ Manifest: {manifest_path}")
    print(f"\n📊 Category breakdown:")
    print(f"   - ArXiv Medium: 6 configs (base)")
    print(f"   - ArXiv Long: 6 configs (base)")
    print(f"   - ArXiv Extra Long: 9 configs (base + extra)")
    print(f"   - LongForm Medium: 6 configs (base)")
    print(f"   - LongForm Long: 6 configs (base)")
    print(f"   - LongForm Extra Long: 9 configs (base + extra)")
    print(f"\n🎯 Usage:")
    print(f"   python main.py --config <config.yml>  # default attention")
    print(f"   python main.py --config <config.yml> --attention flash_attention_2")
    
    return count

if __name__ == "__main__":
    total = generate_configs()
    print(f"\n🎯 Toplam {total} config hazır!")

