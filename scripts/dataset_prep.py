# scripts/dataset_prep.py
import argparse, json, os, random
from datasets import load_dataset

random.seed(42)

SPLIT_CANDIDATES = ["test", "validation", "valid", "val", "dev", "train"]
# olası alan adları:
TEXT_KEYS = ["document", "text", "article", "body", "content"]
REF_KEYS  = ["summary", "abstract", "target", "reference"]

def pick_split(ds_dict):
    # load_dataset(...) DatasetDict ise uygun split'i seç
    for k in SPLIT_CANDIDATES:
        if k in ds_dict:
            return k
    # değilse None dön
    return None

def pick_fields(example_keys):
    ek = set(example_keys)
    text_key = next((k for k in TEXT_KEYS if k in ek), None)
    ref_key  = next((k for k in REF_KEYS  if k in ek), None)
    if not text_key or not ref_key:
        raise KeyError(f"Alanlar bulunamadı. Mevcut anahtarlar: {sorted(ek)}")
    return text_key, ref_key

def dump_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def prep(dataset_id, out_path, n):
    ds = load_dataset(dataset_id)              # DatasetDict bekliyoruz
    split_name = pick_split(ds)
    split = ds[split_name] if split_name else ds
    # bir örnek alıp alan adlarını keşfet
    first = split[0]
    text_key, ref_key = pick_fields(first.keys())
    N = min(n, len(split))
    rows = []
    for i, ex in enumerate(random.sample(list(split), N)):
        rows.append({
            "id": f"{os.path.basename(out_path).replace('.jsonl','')}_{i}",
            "text": ex[text_key],
            "ref":  ex[ref_key],
        })
    dump_jsonl(out_path, rows)
    print(f"✓ Saved {out_path} ({len(rows)} docs)  split={split_name}  fields=({text_key}, {ref_key})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=["longform","arxiv","both"], default="both")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    if args.ds in ("longform","both"):
        prep("vgoldberg/longform_article_summarization", "data/dev_longform.jsonl", args.n)
    if args.ds in ("arxiv","both"):
        prep("ccdv/arxiv-summarization", "data/dev_arxiv.jsonl", args.n)
import os
print("Current working directory:", os.getcwd())
