"""Run one probe with GPU telemetry, bounded runtime, and mandatory online SwanLab."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading
import shutil
import hashlib

ROOT = Path(__file__).resolve().parents[2]
import pynvml as nv
import swanlab

p=argparse.ArgumentParser()
p.add_argument('--name',required=True)
p.add_argument('--length',type=int,default=2048)
p.add_argument('--budget',type=int)
p.add_argument('--batch',type=int,default=1)
p.add_argument('--steps',type=int,default=3)
p.add_argument('--data')
p.add_argument('--timeout',type=int,default=900)
p.add_argument('--options',default='{}')
a=p.parse_args()
out=ROOT/'preparation/logs/pilot'/a.name
if out.exists() and any(out.iterdir()):
    a.name += '-'+time.strftime('%Y%m%d-%H%M%S')
    out=ROOT/'preparation/logs/pilot'/a.name
out.mkdir(parents=True,exist_ok=True)
case=dict(name=a.name,length=a.length,budget=a.budget or a.length,batch=a.batch,steps=a.steps,
          data=str(Path(a.data).resolve()) if a.data else str(ROOT/'preparation/logs/pilot'/f'stress-{a.length}.parquet'),
          output=str(out),**json.loads(a.options))
case['probe_version'] = 2
case['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
with open(case['data'],'rb') as f:
    case['data_sha256'] = hashlib.file_digest(f,'sha256').hexdigest()
source = out/'source'; source.mkdir(exist_ok=True)
for relative in ['recipe/opd/kl_trainer.py','recipe/opd/recompute_kl.py','recipe/opd/config.py',
                 'verl/workers/engine/fsdp/transformer_impl.py','preparation/pilot/train_probe.py']:
    shutil.copyfile(ROOT/relative,source/relative.replace('/','__'))
(source/'working-tree.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=ROOT))
(out/'case.json').write_text(json.dumps(case,indent=2))
run=swanlab.init(project='trd-pro6000-pilot',name=a.name,mode='online',on_online_error='raise',
    config=case,log_dir=str(out/'swanlab'),description='2x RTX PRO 6000 96GB; FSDP2 4B student / 8B teacher; pilot throughput and memory only.')
url=run.url
(out/'swanlab.json').write_text(json.dumps(dict(url=url,id=run.id),indent=2))
(out/'probe_swanlab.json').write_text(json.dumps(dict(url=url,id=run.id),indent=2))
with (ROOT/'HISTORY.md').open('a') as f:
    f.write(f'\n- Pilot `{a.name}`：SwanLab {url}；配置/日志/显存采样：`{out.relative_to(ROOT)}`（结果见该目录 summary.json）。\n')
nv.nvmlInit(); handles=[nv.nvmlDeviceGetHandleByIndex(i) for i in range(2)]
stop=threading.Event(); telemetry=[]
def monitor():
    t0=time.monotonic()
    while not stop.is_set():
        row={'seconds':time.monotonic()-t0,'monotonic':time.monotonic()}
        for i,h in enumerate(handles):
            row[f'gpu{i}_used_gib']=nv.nvmlDeviceGetMemoryInfo(h).used/2**30
            row[f'gpu{i}_util']=nv.nvmlDeviceGetUtilizationRates(h).gpu
            row[f'gpu{i}_power_w']=nv.nvmlDeviceGetPowerUsage(h)/1000
        telemetry.append(row)
        stop.wait(.1)
th=threading.Thread(target=monitor,daemon=True); th.start()
env=os.environ.copy(); env.update(WANDB_MODE='disabled',SWANLAB_MODE='online' if case.get('native_logger') else 'disabled',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false',
    OTEL_SDK_DISABLED='true',PYTHONPATH=str(ROOT))
env['KL_FULL_VOCAB_CHUNK_SIZE']=str(case.get('chunk',512))
cmd=[str(ROOT/'.venv/bin/torchrun'),'--standalone','--nproc-per-node=2',str(ROOT/'preparation/pilot/train_probe.py'),'--case',str(out/'case.json')]
start=time.monotonic(); timed_out=False
with (out/'train.log').open('w') as log:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        rc=proc.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        timed_out=True; os.killpg(proc.pid,signal.SIGTERM)
        try: rc=proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL); rc=proc.wait()
stop.set();th.join();nv.nvmlShutdown()
(out/'telemetry.json').write_text(json.dumps(telemetry))
results=[]
for rank in range(2):
    path=out/f'result-rank{rank}.json'
    if path.exists(): results.append(json.loads(path.read_text()))
summary=dict(case=case,swanlab_url=url,returncode=rc,timed_out=timed_out,wall_seconds=time.monotonic()-start,
    passed=rc==0 and len(results)==2,results=results,
    gpu_peak_gib=[max(r[f'gpu{i}_used_gib'] for r in telemetry) for i in range(2)])
if results and results[0]['metrics'] and 'start_monotonic' in results[0]['metrics'][0]:
    windows=[(m['start_monotonic'],m['start_monotonic']+m['total_seconds']) for m in results[0]['metrics'][1:]]
    samples=[r for r in telemetry if any(start<=r['monotonic']<=end for start,end in windows)]
    if samples:
        summary['steady_gpu_util_percent']=[sum(r[f'gpu{i}_util'] for r in samples)/len(samples) for i in range(2)]
(out/'summary.json').write_text(json.dumps(summary,indent=2))
for i,row in enumerate(telemetry[::10]): swanlab.log({f'telemetry/{k}':v for k,v in row.items()},step=i)
# Aligned rank results: report slowest rank time and sum the unique DP tokens.
if results:
    for i in range(min(len(r['metrics']) for r in results)):
        rows=[r['metrics'][i] for r in results]
        dt=max(r['total_seconds'] for r in rows)
        dp_divisor=case.get('sp',1)
        swanlab.log({'bench/'+k:v for k,v in dict(iteration=i,total_seconds=dt,
            response_tokens_per_second=sum(r['response_tokens'] for r in rows)/dp_divisor/dt,
            model_tokens_per_second=sum(r['student_tokens']+r['teacher_tokens'] for r in rows)/dp_divisor/dt,
            peak_allocated_gib=max(r['allocated_peak_gib'] for r in rows),
            kl_loss=rows[0]['kl_loss'],teacher_seconds=max(r['teacher_seconds'] for r in rows),
            student_seconds=max(r['student_seconds'] for r in rows)).items()})
swanlab.log({'result/passed':int(summary['passed']),'result/wall_seconds':summary['wall_seconds'],
    'result/gpu0_peak_gib':summary['gpu_peak_gib'][0],'result/gpu1_peak_gib':summary['gpu_peak_gib'][1],
    'artifacts/config':swanlab.Text(json.dumps(case)),
    'artifacts/summary':swanlab.Text(json.dumps(summary)),
    'artifacts/train_log':swanlab.Text((out/'train.log').read_text()[-150000:])})
swanlab.finish(state='success' if summary['passed'] else 'crashed',
               error=None if summary['passed'] else (out/'train.log').read_text()[-10000:])
print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2),flush=True)
sys.exit(0 if summary['passed'] else 1)
