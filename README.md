# Long-Context Retrieval for Document Summarization (CENG543)

A research project investigating long-document summarization strategies using sliding window approaches, retrieval-augmented generation, and index compression techniques (FAISS, ColBERT, PLAID).

## Prerequisites

- **Python 3.12** (required)
- **uv** package manager (recommended) - [Installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- **NVIDIA GPU** with CUDA support (recommended for model inference)
- At least **16GB RAM** recommended

## Environment Setup

### Option 1: Using uv (Recommended)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with Python 3.12
uv venv --python 3.12
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
uv pip install -r requirements.txt
```

### Option 2: Using pip

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Post-installation: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

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

# Arxiv dataset
uv run python -u scripts/filter_long_docs.py --in src/data/dev_arxiv.jsonl --out src/data/uncleaned/medium_examples_arxiv.jsonl --min_tokens 4000 --max_tokens 8000
uv run python -u scripts/filter_long_docs.py --in src/data/dev_arxiv.jsonl --out src/data/uncleaned/long_examples_arxiv.jsonl --min_tokens 8000 --max_tokens 16000
uv run python -u scripts/filter_long_docs.py --in src/data/dev_arxiv.jsonl --out src/data/uncleaned/extra_long_examples_arxiv.jsonl --min_tokens 16000 --max_tokens 32000
```

### Step 3: Normalize references

```bash
uv run python scripts/normalize_ref.py
```

---

## Running Experiments

### 1. Sliding Window Experiments

**Single config run:**
```bash
uv run python main.py --config configs/sliding/grid/arxiv_extra_long_w1024_ov256_g16.yml
```

**With model override (BigBird):**
```bash
uv run python main.py --config configs/sliding/grid/arxiv_extra_long_w1024_ov256_g16.yml --model google/bigbird-pegasus-large-arxiv
```

**Run all grid configs (Linux):**
```bash
for cfg in configs/sliding/grid/*.yml; do uv run python main.py --config "$cfg"; done
```

---

### 2. Chunk Retrieval Experiments

Retrieve relevant chunks with dense retrieval before summarization.

**Test chunk retrieval pipeline:**
```bash
uv run python test_chunk_retrieval.py
```

**Run all chunk retrieval configs (Linux):**
```bash
for cfg in configs/chunk_retrieval/*.yml; do uv run python main_chunk_retrieval.py --config "$cfg"; done
```

**Run all chunk retrieval configs (Windows PowerShell):**
```powershell
Get-ChildItem configs/chunk_retrieval/*.yml | ForEach-Object { python main_chunk_retrieval.py --config $_.FullName }
```

---

### 3. FAISS Index Benchmark

Tests different FAISS index types (Flat, IVF, IVF-PQ) for retrieval quality vs. compression trade-offs.

**Run full benchmark (ArXiv):**
```bash
uv run python main_faiss_benchmark_v2.py --config configs/faiss_benchmark/full_grid_v2.yml
```

**Run full benchmark (Longform):**
```bash
uv run python main_faiss_benchmark_v2.py --config configs/faiss_benchmark/full_grid_v2_longform.yml
```

**Available FAISS configs:**
| Config | Description |
|--------|-------------|
| `flat_baseline.yml` | Exact search baseline |
| `ivf_only.yml` | IVF index only |
| `ivf_pq_4bit.yml` | IVF-PQ with 4-bit quantization |
| `ivf_pq_8bit.yml` | IVF-PQ with 8-bit quantization |
| `full_grid_v2.yml` | Full grid search (ArXiv) |
| `full_grid_v2_longform.yml` | Full grid search (Longform) |

---

### 4. ColBERT Benchmark

Token-level late interaction retrieval experiments.

**Run ColBERT benchmark (ArXiv):**
```bash
uv run python main_colbert_benchmark.py --config configs/colbert_benchmark/full_grid.yml
```

**Run ColBERT benchmark (Longform):**
```bash
uv run python main_colbert_benchmark.py --config configs/colbert_benchmark/full_grid_longform.yml
```

---

### 5. PLAID Benchmark

Tests PLAID token pruning + PQ compression for memory-efficient retrieval.

**Run PLAID benchmark (ArXiv):**
```bash
uv run python main_plaid_benchmark.py --config configs/plaid_benchmark/full_grid.yml
```

**Run PLAID benchmark (Longform):**
```bash
uv run python main_plaid_benchmark.py --config configs/plaid_benchmark/full_grid_longform.yml
```

---

### 6. Aggregation Comparison (concat vs hierarchical)

Test if hierarchical summarization improves ROUGE-L compared to simple concatenation.

**Run hierarchical tests:**
```bash
bash scripts/run_aggregation_test.sh
```

**Compare results:**
```bash
uv run python scripts/compare_aggregation.py
```

---

## Analyze Results

```bash
# Grid search analysis
uv run python scripts/analyze_grid_results.py

# Generate paper figures
uv run python scripts/generate_paper_figures.py

# FAISS compression analysis
uv run python scripts/plot_faiss_compression.py

# PLAID analysis plots
uv run python scripts/plot_plaid_analysis.py
```

---

## Project Structure

```
├── main.py                      # Sliding window experiments
├── main_chunk_retrieval.py      # Chunk retrieval experiments
├── main_faiss_benchmark_v2.py   # FAISS index benchmarks
├── main_colbert_benchmark.py    # ColBERT benchmarks
├── main_plaid_benchmark.py      # PLAID benchmarks
├── configs/
│   ├── sliding/                 # Sliding window configs
│   │   └── grid/                # Grid search configurations
│   ├── chunk_retrieval/         # Chunk retrieval configs
│   ├── faiss_benchmark/         # FAISS benchmark configs
│   ├── colbert_benchmark/       # ColBERT configs
│   └── plaid_benchmark/         # PLAID configs
├── src/
│   ├── data/                    # Datasets
│   │   ├── cleaned/             # Processed datasets
│   │   └── uncleaned/           # Raw filtered datasets
│   ├── models/                  # Model implementations
│   │   ├── sliding.py           # Sliding window summarizer
│   │   ├── retriever.py         # Dense retriever
│   │   ├── retriever_advanced.py# Advanced FAISS indices
│   │   ├── colbert_retriever.py # ColBERT implementation
│   │   ├── plaid_retriever.py   # PLAID implementation
│   │   └── generator.py         # Summary generator
│   ├── eval/                    # Evaluation metrics (ROUGE-L, BERTScore)
│   └── utils/                   # Utility functions
├── scripts/                     # Data prep & analysis scripts
├── results/                     # Experiment outputs
└── cache/                       # Cached embeddings & indices
```

---

## Aggregation Methods

- **concat** (default): Concatenate all window summaries with space
- **hierarchical**: Summarize the concatenated summaries for coherence

Set in config YAML:
```yaml
aggregation: hierarchical
```

---

## Cache Management

The project uses caching for embeddings and indices to speed up repeated runs.

```bash
# Clear all cache
rm -rf cache/

# Clear specific benchmark cache
rm -rf cache/faiss_benchmark_v2/
rm -rf cache/colbert_benchmark/
rm -rf cache/plaid_benchmark/
rm -rf cache/embeddings/
```

---

## Troubleshooting

### CUDA out of memory
- Reduce `samples` count in config file
- Use smaller model (`allenai/led-base-16384` instead of `led-large`)
- Reduce `gen_max_tokens` in config

### NLTK data missing
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Flash Attention errors
Set `attention_impl: default` in config YAML or use `--attention default` flag.

### Slow embedding generation
Enable caching in config:
```yaml
use_cache: true
cache_dir: cache/embeddings
```

---

## Citation

If you use this project in your research, please cite accordingly.

## License

This project is for academic research purposes (CENG543 course project).