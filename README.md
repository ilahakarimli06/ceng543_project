# Long-Context Retrieval-Augmented Summarization (CENG543)

Long-document summarization using sliding window approach with LED and BigBird models, combined with retrieval-augmented methods (Dense, ColBERT, PLAID).


## Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| **Longform** | [vgoldberg/longform_article_summarization](https://huggingface.co/datasets/vgoldberg/longform_article_summarization) | Long-form articles with human-written summaries |
| **ArXiv** | [ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization) | Scientific papers with abstracts as summaries |

### Document Length Categories

| Category | Token Range | Description |
|----------|-------------|-------------|
| Medium | 4,000 - 8,000 | 2-4x model context window |
| Long | 8,000 - 16,000 | 4-8x model context window |
| Extra Long | 16,000 - 32,000 | 8-16x model context window |

## Models & Methods

### Summarization Models

| Model | HuggingFace ID | Context Window |
|-------|----------------|----------------|
| LED | `allenai/led-base-16384` | 16,384 tokens |
| BigBird-Pegasus | `google/bigbird-pegasus-large-arxiv` | 4,096 tokens |
| LongT5 | `google/long-t5-tglobal-base` | 16,384 tokens |

### Retrieval Methods

| Method | Description | Implementation |
|--------|-------------|----------------|
| **Dense (FAISS)** | Bi-encoder embeddings with FAISS indexing | Flat, IVF, IVF-PQ variants |
| **ColBERT** | Token-level late interaction with MaxSim scoring | Full token embeddings |
| **PLAID** | Compressed ColBERT with centroid-based pruning | Memory-efficient ColBERT |

### Embedding Models

| Model | HuggingFace ID | Dimension |
|-------|----------------|-----------|
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| BGE | `BAAI/bge-base-en-v1.5` | 768 |

## Hardware & Environment

### Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GPU with CUDA | RTX 3080+ / A100 |
| **VRAM** | 8 GB | 16+ GB |
| **RAM** | 16 GB | 32+ GB |
| **CUDA** | 11.x | 12.x |
| **Python** | 3.10 | 3.11+ |

### Reproducibility Settings

All experiments use fixed random seeds for reproducibility:
- **Seed**: 42 (fixed across all experiments)
- **Samples**: 60 documents per configuration
- See `configs/default.yml` for all default hyperparameters

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/ceng543_project.git
cd ceng543_project

# Install dependencies
uv pip install -r requirements.txt
```

## Dataset Preparation

### Step 1: Download datasets

```bash
uv run python -u scripts/dataset_prep.py --ds both --n 800
```

### Step 2: Filter by document length

```bash
# Longform dataset
uv run python -u scripts/filter_long_docs.py --in src/data/dev_longform.jsonl --out src/data/uncleaned/medium_examples_longform.jsonl --min_tokens 4000 --max_tokens 8000
uv run python -u scripts/filter_long_docs.py --in src/data/dev_longform.jsonl --out src/data/uncleaned/long_examples_longform.jsonl --min_tokens 8000 --max_tokens 16000
uv run python -u scripts/filter_long_docs.py --in src/data/dev_longform.jsonl --out src/data/uncleaned/extra_long_examples_longform.jsonl --min_tokens 16000 --max_tokens 32000

# ArXiv dataset
uv run python -u scripts/filter_long_docs.py --in src/data/dev_arxiv.jsonl --out src/data/uncleaned/medium_examples_arxiv.jsonl --min_tokens 4000 --max_tokens 8000
uv run python -u scripts/filter_long_docs.py --in src/data/dev_arxiv.jsonl --out src/data/uncleaned/long_examples_arxiv.jsonl --min_tokens 8000 --max_tokens 16000
uv run python -u scripts/filter_long_docs.py --in src/data/dev_arxiv.jsonl --out src/data/uncleaned/extra_long_examples_arxiv.jsonl --min_tokens 16000 --max_tokens 32000
```

### Step 3: Normalize references

```bash
uv run python scripts/normalize_ref.py
```

## Reproducing Paper Tables

### Quick Start: Run All Experiments

```bash
# Linux/Mac
bash scripts/reproduce_tables.sh

# Windows PowerShell
.\scripts\reproduce_tables.ps1
```

### Manual Reproduction

#### Table 1-2: Grid Search (Sliding Window)

```bash
# Run all grid search configs
for cfg in configs/sliding/grid/*.yml; do uv run python main.py --config "$cfg"; done

# Analyze results
uv run python scripts/analyze_grid_results.py
```

#### Table 3-4: Chunk Retrieval

```bash
for cfg in configs/chunk_retrieval/*.yml; do uv run python main_chunk_retrieval.py --config "$cfg"; done
```

#### Table 5: FAISS Benchmark

```bash
uv run python main_faiss_benchmark_v2.py --config configs/faiss_benchmark/full_grid_v2.yml
uv run python main_faiss_benchmark_v2.py --config configs/faiss_benchmark/full_grid_v2_longform.yml
```

#### Table 6: ColBERT/PLAID Benchmark

```bash
# ColBERT
uv run python main_colbert_benchmark.py --config configs/colbert_benchmark/full_grid.yml

# PLAID
uv run python main_plaid_benchmark.py --config configs/plaid_benchmark/full_grid.yml
```

#### Equal Budget Comparison

```bash
uv run python generate_equal_budget_report.py --budgets 100 500 1000
```

## Running Individual Experiments

### Single config run

```bash
uv run python main.py --config configs/sliding/grid/arxiv_extra_long_w1024_ov256_g16.yml
```

### With model override (BigBird)

```bash
uv run python main.py --config configs/sliding/grid/arxiv_extra_long_w1024_ov256_g16.yml --model google/bigbird-pegasus-large-arxiv
```

### Test chunk retrieval pipeline

```bash
uv run python test_chunk_retrieval.py
```

## Results Analysis

```bash
uv run python scripts/analyze_grid_results.py
```

## Aggregation Methods

- **concat** (default): Concatenate all window summaries with space
- **hierarchical**: Summarize the concatenated summaries for coherence

Set in config YAML:
```yaml
aggregation: hierarchical
```

## Project Structure

```
ceng543_project/
├── configs/               # Experiment configurations
│   ├── default.yml        # Default hyperparameters
│   ├── sliding/           # Sliding window configs
│   ├── chunk_retrieval/   # Chunk retrieval configs
│   ├── faiss_benchmark/   # FAISS benchmark configs
│   ├── colbert_benchmark/ # ColBERT benchmark configs
│   └── plaid_benchmark/   # PLAID benchmark configs
├── scripts/               # Utility scripts
│   ├── reproduce_tables.sh    # Reproduce all paper tables (Linux/Mac)
│   ├── reproduce_tables.ps1   # Reproduce all paper tables (Windows)
│   └── analyze_grid_results.py
├── src/                   # Source code
│   ├── models/            # Model implementations
│   ├── eval/              # Evaluation metrics
│   └── utils/             # Utility functions
├── results/               # Experiment results
└── requirements.txt       # Python dependencies
```

## Citation

If you use this code, please cite:

@misc{ceng543_long_context,
  title={Long-Context Retrieval-Augmented Summarization},
  author={Ilaha Karimli},
  year={2026},
  institution={Izmir Institute of Technology CENG543}
}