"""Sequential GPU probes; every child creates and uploads an online experiment."""
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
base={'kl_loss_backend':'recompute'}
cases=[
 ('recompute-18k',18432,1,base),
 ('resident-18k',18432,1,dict(base,teacher_cpu_offload=False)),
 ('resident-18k-b2',18432,2,dict(base,teacher_cpu_offload=False)),
 ('resident-18k-b3',18432,3,dict(base,teacher_cpu_offload=False)),
]
for name,length,batch,opts in cases:
    print('START CASE',name,flush=True)
    subprocess.run([sys.executable,str(ROOT/'preparation/pilot/run_probe.py'),'--name',name,
        '--length',str(length),'--budget',str(length*batch),'--batch',str(batch),'--steps','3','--options',json.dumps(opts)],check=False)
