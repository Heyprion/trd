"""Final numerical, memory ceiling and real-data checks, sequentially on the same GPUs."""
from pathlib import Path
import json
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
env=os.environ.copy();env['CHECK_DEVICE']='cuda'
subprocess.run([sys.executable,str(ROOT/'preparation/pilot/check_autocast.py')],env=env,check=True)
opts=dict(kl_loss_backend='recompute',model_storage_dtype='bf16',use_liger=True,teacher_cpu_offload=False,cuda_memory_fraction=.92)
for name,length,batch,steps,extra in [
    ('capped-resident-16k-b2',16384,2,4,dict(accum=2,refine=True)),
    ('capped-resident-2k-b16',2048,16,3,{}),
]:
    subprocess.run([sys.executable,str(ROOT/'preparation/pilot/run_probe.py'),'--name',name,'--length',str(length),'--budget',str(length*batch),'--batch',str(batch),'--steps',str(steps),'--options',json.dumps(dict(opts,**extra))],check=False)
subprocess.run([sys.executable,str(ROOT/'preparation/pilot/rollout_probe.py')],check=True)
summary=json.loads((ROOT/'preparation/logs/pilot/real-rollout/summary.json').read_text())
subprocess.run([sys.executable,str(ROOT/'preparation/pilot/run_probe.py'),'--name','real-opd-trd-update','--length','18432','--budget','32768','--batch','2','--steps','2','--data',summary['training_data'],'--options',json.dumps(dict(opts,refine=True,native_epoch=True,native_logger=True))],check=True)
