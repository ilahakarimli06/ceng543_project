import time, torch
from evaluate import load
rouge = load("rouge")
bertscore = load("bertscore")

def start_prof():
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    return time.time()

def stop_prof(t0):
    import math, torch
    lat = time.time()-t0
    mem = (torch.cuda.max_memory_allocated()/1e9) if torch.cuda.is_available() else 0.0
    return lat, mem

def text_metrics(preds, refs):
    r = rouge.compute(predictions=preds, references=refs)
    b = bertscore.compute(predictions=preds, references=refs, lang="en")
    return {"rougeL": r["rougeL"], "bertscore_f1": sum(b["f1"])/len(b["f1"])}
