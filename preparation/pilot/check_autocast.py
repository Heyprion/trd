"""Check the production autocast context against FP32 loss/gradient references."""
from pathlib import Path
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
import swanlab
from recipe.opd.kl_utils import _forward_kl_chunk,_reverse_kl_chunk
from recipe.opd.recompute_kl import recompute_kl
device=torch.device(os.environ.get('CHECK_DEVICE','cpu'))
out=ROOT/f'preparation/logs/pilot/autocast-numerical-{device.type}';out.mkdir(parents=True,exist_ok=True)
run=swanlab.init(project='trd-pro6000-pilot',name=f'autocast-full-vocab-fp64-reference-{device.type}',mode='online',on_online_error='raise',log_dir=str(out/'swanlab'),config={'device':str(device),'autocast':'bf16','vocab':151936,'reference':'stable FP64 log_softmax, autocast disabled'})
with (ROOT/'HISTORY.md').open('a') as f:f.write(f'\n- 生产 autocast 环境全词表数值/梯度检验：{run.url}；`{out.relative_to(ROOT)}`。\n')
torch.set_num_threads(4);rows=[]
try:
    for near in [False,True]:
        for reverse in [False,True]:
            torch.manual_seed(17)
            s=torch.randn(9,151936,dtype=torch.bfloat16,device=device,requires_grad=True)
            t=(s.detach()+.02*torch.randn_like(s)) if near else torch.randn_like(s)
            si=torch.tensor([0,2,4,5,8],device=device);ti=si
            fn=_reverse_kl_chunk if reverse else lambda s,t:_forward_kl_chunk(t,s)
            with torch.autocast(device.type,enabled=False):
                lp=s[si].double().log_softmax(-1);lq=t[ti].double().log_softmax(-1)
                ref=((lp.exp()*(lp-lq)) if reverse else (lq.exp()*(lq-lp))).sum(-1)
            with torch.autocast(device.type,dtype=torch.bfloat16):
                old=fn(s[si],t[ti]);fast=recompute_kl(s,t,si,ti,chunk_size=2,reverse=reverse)
            rg=torch.autograd.grad(ref.sum(),s,retain_graph=True)[0].float()
            og=torch.autograd.grad(old.sum(),s,retain_graph=True)[0].float()
            fg=torch.autograd.grad(fast.sum(),s)[0].float()
            row=dict(near_identical=near,reverse=reverse,
                fast_value_max_error=float((fast-ref).abs().max().detach()),
                legacy_value_max_error=float((old-ref).abs().max().detach()),
                fast_gradient_relative_error=float((fg-rg).norm()/rg.norm()),
                legacy_gradient_relative_error=float((og-rg).norm()/rg.norm()))
            assert row['fast_value_max_error']<1e-5,row
            assert row['fast_gradient_relative_error']<.008,row
            rows.append(row)
    (out/'results.json').write_text(json.dumps(rows,indent=2));swanlab.log({'checks/passed':len(rows),'results':swanlab.Text(json.dumps(rows))});swanlab.finish();print(json.dumps(rows,indent=2))
except Exception:
    import traceback
    swanlab.log({'error':swanlab.Text(traceback.format_exc())});swanlab.finish(state='crashed');raise
