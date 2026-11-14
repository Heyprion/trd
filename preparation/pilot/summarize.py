"""Produce a compact, reproducible comparison from the raw rank and NVML results."""
from pathlib import Path
import json
import statistics
ROOT=Path(__file__).resolve().parents[2]
rows=[]
for path in sorted((ROOT/'preparation/logs/pilot').glob('*/summary.json')):
    x=json.loads(path.read_text());c=x.get('case')
    if c is None:continue
    r=dict(name=c['name'],passed=x['passed'],length=c['length'],batch=c['batch'],budget=c['budget'],gpu_peak_gib=max(x['gpu_peak_gib']),url=x['swanlab_url'])
    if x['passed']:
        ranks=x['results']; n=min(len(r['metrics']) for r in ranks)
        timed=[];tok=[];modeltok=[]
        # Drop initialization/warmup; compare the same number of synchronized steps.
        for i in range(1,n):
            ms=[rr['metrics'][i] for rr in ranks];dt=max(m['total_seconds'] for m in ms)
            timed.append(dt)
            tok.append(sum(m['response_tokens'] for m in ms)/c.get('sp',1)/dt)
            modeltok.append(sum(m['student_tokens']+m['teacher_tokens'] for m in ms)/c.get('sp',1)/dt)
        if timed:r.update(seconds=statistics.mean(timed),response_tokens_per_second=statistics.mean(tok),model_tokens_per_second=statistics.mean(modeltok),max_allocated_gib=max(m['allocated_peak_gib'] for rr in ranks for m in rr['metrics']))
    rows.append(r)
report=ROOT/'preparation/pilot/results.json';report.write_text(json.dumps(rows,indent=2))
print('| case | pass | per-GPU tokens | batch | NVML peak GiB | step sec | response tok/s |')
print('|---|---|---:|---:|---:|---:|---:|')
for r in rows:print(f"| {r['name']} | {r['passed']} | {r['budget']} | {r['batch']} | {r['gpu_peak_gib']:.2f} | {r.get('seconds',0):.3f} | {r.get('response_tokens_per_second',0):.0f} |")
