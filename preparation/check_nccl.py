#!/usr/bin/env python3
"""Run with torchrun --standalone --nproc-per-node=2 preparation/check_nccl.py."""
import os
import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
value = torch.tensor([dist.get_rank() + 1.0], device=f"cuda:{local_rank}")
dist.all_reduce(value)
expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
assert value.item() == expected, (value.item(), expected)
print(f"Rank {dist.get_rank()}: NCCL all_reduce passed ({value.item()})", flush=True)
dist.destroy_process_group()
