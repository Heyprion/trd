"""Publish the final comparison and recommended profiles alongside the individual runs."""
from pathlib import Path
import json
import swanlab
ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'preparation/pilot'
validation=json.loads((folder/'validation.json').read_text())
rows={r['name']:r for r in json.loads((folder/'results.json').read_text())}
run=swanlab.init(project='trd-pro6000-pilot',name='pilot-final-comparison',mode='online',on_online_error='raise',
    log_dir=str(ROOT/'preparation/logs/pilot/overview-swanlab'),job_type='summary',
    config={'gpu':'2x RTX PRO 6000 Blackwell 96GB','training_probes':21,'passed':17,'exploratory_oom':4,
            'fsdp_size':2,'sp':1,'recommended_global_batch':128,'recommended_gradient_accumulation':32,
            'tested_long_gradient_accumulation':[2,4],'quality_evaluation':False})
url=run.url
(folder/'overview.json').write_text(json.dumps({'url':url},indent=2)+'\n')
with (folder/'REPORT.zh-CN.md').open('a') as f:f.write(f'\nSwanLab [首轮结果汇总与推荐配置]({url})。\n')
with (ROOT/'HISTORY.md').open('a') as f:f.write(f'\n- 首轮完整报告、候选对比与两套推荐配置已上传 SwanLab 汇总：{url}。\n')
metrics={}
for label,name in [('balanced_long','dynamic32k-resident-18k-b2'),('balanced_full_budget','capped-resident-16k-b2'),('high_memory','capped-offload-18k-b2-acc4'),('short','capped-resident-2k-b16')]:
    row=rows[name];metrics[f'{label}/nvml_peak_gib']=row['gpu_peak_gib'];metrics[f'{label}/response_tokens_per_second']=row['response_tokens_per_second']
metrics.update({'artifacts/report':swanlab.Text((folder/'REPORT.zh-CN.md').read_text()),
    'artifacts/results':swanlab.Text((folder/'results.json').read_text()),
    'artifacts/validation':swanlab.Text(json.dumps(validation)),
    'artifacts/balanced_profile':swanlab.Text((folder/'profiles/balanced.env.sh').read_text()),
    'artifacts/high_memory_profile':swanlab.Text((folder/'profiles/high_memory.env.sh').read_text())})
swanlab.log(metrics);swanlab.finish();print(url)
