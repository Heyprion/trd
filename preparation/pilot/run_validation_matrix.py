from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
base=dict(kl_loss_backend='recompute',model_storage_dtype='bf16',use_liger=True)
cases=[
 ('capped-offload-18k-b2-acc4',18432,36864,2,8,dict(base,teacher_cpu_offload=True,cuda_memory_fraction=.92,accum=4,refine=True)),
 ('dynamic32k-resident-18k-b2',18432,32768,2,4,dict(base,teacher_cpu_offload=False,cuda_memory_fraction=.92,refine=True)),
 ('sp2-resident-18k-b1',18432,18432,1,3,dict(base,teacher_cpu_offload=False,sp=2)),
]
for name,length,budget,batch,steps,opts in cases:
    print('START CASE',name,flush=True)
    subprocess.run([sys.executable,str(ROOT/'preparation/pilot/run_probe.py'),'--name',name,
        '--length',str(length),'--budget',str(budget),'--batch',str(batch),'--steps',str(steps),'--options',json.dumps(opts)],check=False)
