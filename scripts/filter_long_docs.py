# scripts/filter_long_docs.py
import argparse, json, sys
from pathlib import Path
from transformers import AutoTokenizer

TEXT_KEYS = ["text", "document", "article", "body", "content"]
REF_KEYS  = ["ref", "summary", "abstract", "target", "reference"]

def get_first_key(d: dict, keys):
    for k in keys:
        if k in d and isinstance(d[k], str) and d[k].strip():
            return k
    return None

def filter_by_tokens(in_path: Path, out_path: Path, min_tokens: int, max_tokens, include_ref: bool):
    tok = AutoTokenizer.from_pretrained("allenai/led-base-16384")
    in_path = in_path.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[ERR] Input not found: {in_path}", file=sys.stderr)
        return 1

    total = kept = skipped = 0
    print(f"[INFO] Reading : {in_path}")
    print(f"[INFO] Writing : {out_path}")
    max_str = "inf" if max_tokens is None else str(max_tokens)
    print(f"[INFO] Threshold: min_tokens={min_tokens}  max_tokens={max_str}  include_ref={include_ref}")

    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            try:
                ex = json.loads(line)
            except Exception:
                skipped += 1
                continue

            # metin ve (opsiyonel) referans alanlarını bul
            tkey = get_first_key(ex, TEXT_KEYS)
            if not tkey:
                skipped += 1
                continue
            text = ex[tkey]

            if include_ref:
                rkey = get_first_key(ex, REF_KEYS)
                if rkey:
                    text = text + " " + ex[rkey]

            # sadece uzunluğu ölç (kesme yok)
            try:
                n = len(tok(text, truncation=False)["input_ids"])
            except Exception:
                skipped += 1
                continue

            if n >= min_tokens and (max_tokens is None or n < max_tokens):
                ex["n_tokens"] = n
                ex["_length_source"] = "text+ref" if include_ref else "text"
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept += 1

    max_str = "inf" if max_tokens is None else str(max_tokens)
    print(f"[DONE] Kept {kept}/{total} docs ({min_tokens} <= tokens < {max_str}) | skipped(empty/bad)={skipped}")
    print(f"[DONE] Saved → {out_path}")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Filter JSONL docs by token length")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL path")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL path")
    ap.add_argument("--min_tokens", type=int, default=0, help="Minimum token count (inclusive)")
    ap.add_argument("--max_tokens", type=int, default=None, help="Maximum token count (exclusive). Default: no limit")
    ap.add_argument("--include_ref", action="store_true",
                    help="Also count reference/summary tokens (text+ref). Default: only text.")
    args = ap.parse_args()

    sys.exit(filter_by_tokens(Path(args.inp), Path(args.out), args.min_tokens, args.max_tokens, args.include_ref))
