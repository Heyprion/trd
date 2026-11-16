"""Measured real FSDP2 OPD updates. Launched by run_probe.py, which owns SwanLab."""
import argparse
from collections import Counter
import dataclasses
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['WANDB_MODE'] = 'disabled'
import torch
import torch.distributed as dist
from recipe.opd.config import KLTrainingConfig
from recipe.opd.kl_trainer import KLTrainer

p = argparse.ArgumentParser()
p.add_argument('--case', required=True)
a = p.parse_args()
case = json.loads(Path(a.case).read_text())
out = Path(case['output'])
rank = int(os.environ['RANK'])
trainer = None
try:
    cfg = KLTrainingConfig(
        distill_mode='opd', task='math', kl_type=case.get('kl_type', 'forward'), kl_method='full_vocab',
        temperature=1.0, kl_token_clip=0,
        student_model_path=str(ROOT/'models/Qwen3-4B-Instruct-2507'),
        teacher_model_path=str(ROOT/'models/Qwen3-8B'),
        use_lora=True, lora_rank=64, lora_alpha=128, learning_rate=5e-6,
        train_batch_size=case.get('batch',1), gradient_accumulation_steps=case.get('accum',1),
        max_length=case['length'], max_token_len_per_gpu=case['budget'],
        use_initial_response=case.get('refine',False), prompt_truncation=True,
        sequence_format=case.get('sequence_format','plain'),
        data_path=case['data'], num_workers=0, n_gpus_per_node=2,
        fsdp_strategy='fsdp2', fsdp_size=2, ulysses_sequence_parallel_size=case.get('sp',1),
        use_torch_compile=case.get('compile',False), gradient_checkpointing=case.get('checkpoint',True),
        save_steps=100000, save_merged_model=False, run_eval_after_training=False,
        output_dir=str(out), model_save_dir=str(out), logging_steps=1,
        wandb_run_name=case['name'],
    )
    # Later optimizations are explicit and optional; old trainer accepts the baseline.
    for key in ('teacher_cpu_offload','reshard_after_forward','kl_loss_backend','use_liger','model_storage_dtype','cuda_memory_fraction'):
        if key in case:
            setattr(cfg, key, case[key])
    started = time.perf_counter()
    trainer = KLTrainer(cfg)
    dtypes = {}
    for label, engine in [('student',trainer.student_engine),('teacher',trainer.teacher_engine)]:
        counts = Counter()
        for param in engine.module.parameters():
            local = param.to_local() if hasattr(param,'to_local') else param
            counts[f'{local.dtype}_trainable={param.requires_grad}'] += local.numel()
        dtypes[label] = dict(counts)
    (out/f'dtypes-rank{rank}.json').write_text(json.dumps(dtypes,indent=2))
    (out/f'config-rank{rank}.json').write_text(json.dumps(dataclasses.asdict(cfg),indent=2))
    init_seconds = time.perf_counter()-started
    trainer.student_engine.optimizer_zero_grad()
    measured = []
    with trainer.student_engine.train_mode():
        for i, batch in enumerate(trainer.train_dataloader):
            if i >= case.get('steps',3):
                break
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ts = time.perf_counter()
            tb = trainer._make_teacher_batch(batch)
            with trainer.teacher_engine.eval_mode():
                teacher = trainer.teacher_engine.forward_backward_batch(tb, loss_function=None, forward_only=True)
            torch.cuda.synchronize()
            teacher_sec = time.perf_counter()-ts
            sb = trainer._make_student_batch(batch, teacher['model_output'])
            ts2 = time.perf_counter()
            student = trainer.student_engine.forward_backward_batch(sb, loss_function=trainer._compute_kl_loss, forward_only=False)
            torch.cuda.synchronize()
            student_sec = time.perf_counter()-ts2
            ts3 = time.perf_counter()
            grad_norm = None
            if (i+1) % cfg.gradient_accumulation_steps == 0 or i+1 == case.get('steps',3):
                grad_norm = float(trainer.student_engine.optimizer_step())
                assert math.isfinite(grad_norm) and grad_norm > 0, grad_norm
                trainer.student_engine.optimizer_zero_grad()
                trainer.student_engine.lr_scheduler_step()
                trainer.global_step += 1
            torch.cuda.synchronize()
            total_sec = time.perf_counter()-ts
            m = trainer._summarize_step_output(student)
            assert all(math.isfinite(v) for v in m.values()), m
            row = dict(iteration=i, start_monotonic=ts, teacher_seconds=teacher_sec, student_seconds=student_sec,
                optimizer_seconds=time.perf_counter()-ts3, total_seconds=total_sec, grad_norm=grad_norm,
                allocated_peak_gib=torch.cuda.max_memory_allocated()/2**30,
                reserved_peak_gib=torch.cuda.max_memory_reserved()/2**30,
                student_tokens=int(batch['student_input_ids'].values().numel()),
                teacher_tokens=int(batch['teacher_input_ids'].values().numel()), **m)
            measured.append(row)
            (out/f'metrics-rank{rank}.json').write_text(json.dumps(measured,indent=2))
            if rank == 0:
                print('PROBE_METRIC '+json.dumps(row),flush=True)
            del sb, tb, teacher, student, batch
    native_epoch_metrics = None
    if case.get('native_epoch'):
        before = trainer.global_step
        native_epoch_metrics = trainer.train_epoch()
        assert trainer.global_step > before
        assert all(math.isfinite(v) for v in native_epoch_metrics.values())
    (out/f'result-rank{rank}.json').write_text(json.dumps(dict(status='passed',init_seconds=init_seconds,optimizer_steps=trainer.global_step,metrics=measured,native_epoch_metrics=native_epoch_metrics),indent=2))
    trainer.close()
except BaseException as e:
    import traceback
    traceback.print_exc()
    (out/f'error-rank{rank}.json').write_text(json.dumps(dict(status='failed',error=repr(e),traceback=traceback.format_exc()),indent=2))
    sys.stdout.flush(); sys.stderr.flush()
    # Avoid a destructor barrier if one rank encounters OOM.
    os._exit(1)
