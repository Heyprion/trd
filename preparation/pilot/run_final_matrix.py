from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
base=dict(kl_loss_backend='recompute',model_storage_dtype='bf16',use_liger=True)
cases=[
 ('offload-liger-18k-b2',18432,2,dict(base,teacher_cpu_offload=True)),
 ('resident-liger-18k-b1',18432,1,dict(base,teacher_cpu_offload=False)),
 ('resident-liger-16k-b2',16384,2,dict(base,teacher_cpu_offload=False)),
 ('resident-liger-2k-b16',2048,16,dict(base,teacher_cpu_offload=False)),
]
for name,length,batch,opts in cases:
    print('START CASE',name,flush=True)
    subprocess.run([sys.executable,str(ROOT/'preparation/pilot/run_probe.py'),'--name',name,
        '--length',str(length),'--budget',str(length*batch),'--batch',str(batch),'--steps','3','--options',json.dumps(opts)],check=False)
