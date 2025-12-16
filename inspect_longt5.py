import json

# Load LongT5 samples
with open('results/grid/arxiv_extra_long/led_w1024_ov0_g0_longt5_samples.json') as f:
    data = json.load(f)

print('=== LongT5 SAMPLE INSPECTION ===\n')
print(f'Total samples: {len(data["samples"])}')
print(f'\nConfig: {data["config"]["model"]}\n')

print('First 5 samples:\n')
for i, s in enumerate(data['samples'][:5]):
    print(f'--- Sample {i+1} (ID: {s["sample_id"]}) ---')
    print(f'Source length: {s["source_length"]:,} chars')
    print(f'Generated length: {len(s["generated"]):,} chars')
    print(f'Reference length: {len(s["reference"]):,} chars')
    print(f'ROUGE-L: {s["metrics"]["rougeL"]:.6f}')
    print(f'BERTScore: {s["metrics"]["bertscore_f1"]:.4f}')
    print(f'\nGenerated (first 300 chars):')
    print(f'{s["generated"][:300]}...')
    print(f'\nReference (first 300 chars):')
    print(f'{s["reference"][:300]}...')
    print('\n' + '='*60 + '\n')

# Statistics
gen_lengths = [len(s['generated']) for s in data['samples']]
print('\n=== STATISTICS ===')
print(f'Generated summary lengths:')
print(f'  Mean: {sum(gen_lengths)/len(gen_lengths):.1f} chars')
print(f'  Min: {min(gen_lengths)} chars')
print(f'  Max: {max(gen_lengths)} chars')
print(f'\nEmpty summaries (< 10 chars): {sum(1 for l in gen_lengths if l < 10)}')

# Best sample
best = max(data['samples'], key=lambda x: x['metrics']['rougeL'])
print(f'\nSample with highest ROUGE-L:')
print(f'  ID: {best["sample_id"]}')
print(f'  ROUGE-L: {best["metrics"]["rougeL"]:.6f}')
print(f'  Generated: {best["generated"][:500]}')
