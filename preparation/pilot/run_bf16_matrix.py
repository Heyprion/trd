from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
base=dict(kl_loss_backend='recompute',teacher_cpu_offload=False,model_storage_dtype='bf16')
cases=[
 ('bf16-18k-b2',18432,2,base),
 ('bf16-18k-b2-noreshard',18432,2,dict(base,reshard_after_forward=False)),
 ('bf16-18k-b2-liger',18432,2,dict(base,use_liger=True)),
 ('bf16-18k-no-gc',18432,1,dict(base,checkpoint=False)),
 ('bf16-2k-b8',2048,8,base),
]
for name,length,batch,opts in cases:
    print('START CASE',name,flush=True)
    subprocess.run([sys.executable,str(ROOT/'preparation/pilot/run_probe.py'),'--name',name,
        '--length',str(length),'--budget',str(length*batch),'--batch',str(batch),'--steps','3','--options',json.dumps(opts)],check=False)
