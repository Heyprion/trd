"""Exact full-vocabulary KL with bounded FP32 workspace and recomputed backward.

Like chunked distillation losses in Liger Kernel, discard softmax intermediates
and recompute them during backward. Keep the existing LM head and FSDP wrapping.
Only the student's logits need gradients (teacher is frozen). No top-k approximation.
"""
import torch
from torch.autograd.function import once_differentiable


class _RecomputeKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student, teacher, student_indices, teacher_indices, chunk_size, temperature, reverse):
        ctx.save_for_backward(student, teacher, student_indices, teacher_indices)
        ctx.chunk_size, ctx.temperature, ctx.reverse = chunk_size, temperature, reverse
        values = torch.empty(student_indices.numel(), device=student.device, dtype=torch.float32)
        for start in range(0, values.numel(), chunk_size):
            stop = min(start + chunk_size, values.numel())
            s = (student.index_select(0, student_indices[start:stop]) / temperature).float()
            t = (teacher.index_select(0, teacher_indices[start:stop]) / temperature).float()
            logp = s.log_softmax(-1)
            logq = t.log_softmax(-1)
            if reverse:
                values[start:stop] = (logp.exp() * (logp-logq)).sum(-1)
            else:
                values[start:stop] = (logq.exp() * (logq-logp)).sum(-1)
        return values

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        student, teacher, si, ti = ctx.saved_tensors
        grad = torch.zeros_like(student)
        for start in range(0, si.numel(), ctx.chunk_size):
            stop = min(start + ctx.chunk_size, si.numel())
            s = (student.index_select(0, si[start:stop]) / ctx.temperature).float()
            t = (teacher.index_select(0, ti[start:stop]) / ctx.temperature).float()
            if ctx.reverse:
                logp, logq = s.log_softmax(-1), t.log_softmax(-1)
                prob = logp.exp()
                delta = logp-logq
                g = prob * (delta-(prob*delta).sum(-1, keepdim=True))
            else:
                g = s.softmax(-1)-t.softmax(-1)
            g.mul_(grad_output[start:stop,None] / ctx.temperature)
            grad.index_copy_(0, si[start:stop], g.to(grad.dtype))
        return grad, None, None, None, None, None, None


def recompute_kl(student, teacher, student_indices, teacher_indices, *, chunk_size=512, temperature=1., reverse=False):
    if chunk_size < 1 or temperature <= 0:
        raise ValueError('chunk_size and temperature must be positive')
    return _RecomputeKL.apply(student, teacher, student_indices, teacher_indices, chunk_size, temperature, reverse)
