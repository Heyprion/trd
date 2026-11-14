"""Real eight-problem OPD/TRD generation, through the repository's Ray/vLLM pipeline."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'recipe/opd/run/utils'))
import pandas as pd
import pynvml as nv
import swanlab
from datasets import load_from_disk
from transformers import AutoTokenizer
out=ROOT/'preparation/logs/pilot/real-rollout';out.mkdir(parents=True,exist_ok=True)
d=load_from_disk(str(ROOT/'data/train/DeepScaleR'))
tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-4B-Instruct-2507')
# Select real problem-length extremes from a fixed 512-row pool.
lengths=[len(tok.encode(x,add_special_tokens=False)) for x in d[:512]['problem']]
order=sorted((i for i in range(512) if lengths[i]<=1800),key=lambda i:lengths[i]); selected=order[:4]+order[-4:]
d.select(selected).save_to_disk(str(out/'problems'))
opts=dict(MAX_SAMPLES='8',TRAIN_DATA_PATH=str(out/'problems'),
    BASE_PROMPT_LENGTH='2048',MAX_RESPONSE_LENGTH='1024',EXPERT_SOLUTION_PROMPT_LENGTH='4096',
    MAX_PROMPT_LENGTH='2048',Y_R_PROMPT_LENGTH='18432',MAX_LENGTH='18432',
    Y_MODE='y_r',TEACHER_TRAINING_PROMPT='refine',KL_TYPE='forward',KL_METHOD='full_vocab',
    GEN_RESULTS_ROOT=str(out/'generation'),RUN_DATE='pilot-real',
    ROLLOUT_GPU_MEMORY_UTILIZATION='0.85',ROLLOUT_MAX_NUM_SEQS='16',ROLLOUT_MAX_NUM_BATCHED_TOKENS='34816',
    SWANLAB_MODE='disabled')
os.environ.update(opts)
from run_config import build_env
values=build_env()
config=dict(**opts,selected_dataset_indices=selected,problem_tokens=[lengths[i] for i in selected],
    generation_temperature=.6,top_p=.95,top_k=20,tensor_parallel=1,data_parallel=2,
    note='Response cap 1024 for pilot only; generation and rewrite from actual student/teacher.')
(out/'config.json').write_text(json.dumps(config,indent=2))
run=swanlab.init(project='trd-pro6000-pilot',name='real-rollout-dp2-tp1',mode='online',on_online_error='raise',log_dir=str(out/'swanlab'),config=config)
with (ROOT/'HISTORY.md').open('a') as f:f.write(f'\n- 真实 OPD/TRD rollout：{run.url}；`preparation/logs/pilot/real-rollout`。\n')
stop=threading.Event();telemetry=[];nv.nvmlInit();handles=[nv.nvmlDeviceGetHandleByIndex(i) for i in range(2)]
def monitor():
    while not stop.is_set():
        telemetry.append([nv.nvmlDeviceGetMemoryInfo(h).used/2**30 for h in handles]);stop.wait(.1)
th=threading.Thread(target=monitor,daemon=True);th.start();start=time.perf_counter()
with (out/'generation.log').open('w') as f:
    rc=subprocess.call([sys.executable,str(ROOT/'recipe/opd/run/prepare_training_data.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
stop.set();th.join();nv.nvmlShutdown()
summary=dict(returncode=rc,wall_seconds=time.perf_counter()-start,gpu_peak_gib=[max(r[i] for r in telemetry) for i in range(2)],swanlab_url=run.url)
if rc==0:
    path=Path(values['TRAINING_DATA_PATH']); frame=pd.read_parquet(path)
    response_lengths=[len(tok.encode(r[0],add_special_tokens=False)) for r in frame.responses]
    summary.update(training_data=str(path),rows=len(frame),rewritten_response_tokens=response_lengths)
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'telemetry.json').write_text(json.dumps(telemetry))
swanlab.log({'result/success':int(rc==0),'result/wall_seconds':summary['wall_seconds'],
    'result/gpu0_peak_gib':summary['gpu_peak_gib'][0],'result/gpu1_peak_gib':summary['gpu_peak_gib'][1],
    'artifacts/summary':swanlab.Text(json.dumps(summary)),
    'artifacts/log':swanlab.Text((out/'generation.log').read_text()[-180000:])})
swanlab.finish(state='success' if rc==0 else 'crashed');print(json.dumps(summary,indent=2));sys.exit(rc)
