import json, random
def read_jsonl(p): 
    with open(p, "r", encoding="utf-8") as f:
        for line in f: 
            if line.strip(): yield json.loads(line)
def sample_records(path, n, seed=42):
    xs = list(read_jsonl(path))
    random.Random(seed).shuffle(xs)
    return xs[:n]
