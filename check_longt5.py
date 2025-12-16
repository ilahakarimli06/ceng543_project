import json

# Load LongT5 samples
with open('results/grid/arxiv_extra_long/led_w1024_ov0_g0_longt5_samples.json') as f:
    data = json.load(f)

print('=== LongT5 DIAGNOSIS ===\n')

# Check first sample in detail
s = data['samples'][0]
print(f'Sample 0:')
print(f'  Generated: "{s["generated"]}"')
print(f'  Length: {len(s["generated"])} chars')
print(f'  ROUGE-L: {s["metrics"]["rougeL"]}')
print(f'  BERTScore: {s["metrics"]["bertscore_f1"]}')
print(f'\n  Reference (first 200): {s["reference"][:200]}')

# Check all samples
print(f'\n\nAll {len(data["samples"])} samples:')
for i, s in enumerate(data['samples']):
    print(f'{i}: len={len(s["generated"]):4d}, ROUGE={s["metrics"]["rougeL"]:.6f}, text="{s["generated"][:50]}"')
