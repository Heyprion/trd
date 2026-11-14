"""Loss/gradient agreement, including distinct masks, temperatures and clipping."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import swanlab
import torch
from recipe.opd.recompute_kl import recompute_kl
from recipe.opd.kl_utils import _forward_kl_chunk,_reverse_kl_chunk
out=ROOT/'preparation/logs/pilot/numerical';out.mkdir(parents=True,exist_ok=True)
run=swanlab.init(project='trd-pro6000-pilot',name='recompute-numerical-exact-temperature',mode='online',on_online_error='raise',log_dir=str(out/'swanlab'),config={'device':'cpu','reference':'original KL chunk functions','dtypes':['float32','bfloat16']})
with (ROOT/'HISTORY.md').open('a') as f: f.write(f'\n- KL 数值/梯度一致性检验：{run.url}；`preparation/logs/pilot/numerical`。\n')
rows=[]
try:
    torch.set_num_threads(4)
    for dtype in [torch.float32,torch.bfloat16]:
        for reverse in [False,True]:
            for temperature in [.7,1.,1.5]:
                for clip in [0.,.1]:
                    torch.manual_seed(7)
                    s=torch.randn(17,1091,dtype=dtype,requires_grad=True)
                    t=torch.randn(23,1091,dtype=dtype)
                    si=torch.tensor([0,2,4,7,8,9,10,13,16]);ti=si+3
                    fn=_reverse_kl_chunk if reverse else lambda s,t:_forward_kl_chunk(t,s)
                    ref=fn(s.index_select(0,si)/temperature,t.index_select(0,ti)/temperature)
                    fast=recompute_kl(s,t,si,ti,chunk_size=4,temperature=temperature,reverse=reverse)
                    weight=torch.linspace(.1,1,si.numel())
                    rloss=((ref.clamp(max=clip) if clip else ref)*weight).sum()
                    floss=((fast.clamp(max=clip) if clip else fast)*weight).sum()
                    rg=torch.autograd.grad(rloss,s,retain_graph=True)[0].float()
                    fg=torch.autograd.grad(floss,s)[0].float()
                    value_error=float((ref-fast).abs().max()); grad_error=float((rg-fg).norm()/rg.norm().clamp(min=1e-12))
                    assert value_error<2e-5,(dtype,reverse,temperature,clip,value_error)
                    assert grad_error < (0.008 if dtype==torch.bfloat16 else 2e-5),(dtype,reverse,temperature,clip,grad_error)
                    rows.append(dict(dtype=str(dtype),reverse=reverse,temperature=temperature,clip=clip,value_error=value_error,gradient_relative_error=grad_error))
    (out/'results.json').write_text(json.dumps(rows,indent=2))
    swanlab.log({'checks/passed':len(rows),'checks/max_value_error':max(r['value_error'] for r in rows),'checks/max_gradient_relative_error':max(r['gradient_relative_error'] for r in rows),'artifacts/results':swanlab.Text(json.dumps(rows))})
    swanlab.finish();print('Passed',len(rows),'numerical comparisons')
except Exception:
    import traceback
    swanlab.log({'checks/passed':0,'error':swanlab.Text(traceback.format_exc())});swanlab.finish(state='crashed');raise
