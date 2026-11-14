"""Deterministic real math text, repeated only for fixed-length memory stress (not OPD quality)."""
from pathlib import Path
import json
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'preparation/logs/pilot';out.mkdir(parents=True,exist_ok=True)
tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-4B-Instruct-2507')
d=load_from_disk(str(ROOT/'data/train/DeepScaleR'))
for length in [512,2048,4096,8192,12288,16384,18432,24576,32768]:
    rows=[]
    for i in range(128):
        x=d[i % 32]; ids=tok.encode(x['solution'],add_special_tokens=False)
        ids=(ids*((length+len(ids)-1)//len(ids)))[:length]
        rows.append(dict(responses=[tok.decode(ids)],extra_info=dict(problem=x['problem'],answer=x['answer'],expert_cot=x['solution'],initial_response=x['solution'],index=i)))
    pd.DataFrame(rows).to_parquet(out/f'stress-{length}.parquet')
print('Prepared 128 deterministic stress rows (32 unique source problems) per length. Responses are reference text repeated to force truncation at the configured length; no quality conclusions.')
