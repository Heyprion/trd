# Agent 工作约定

## 1. 目的
复现他人的项目，并可能在其他项目上开展探索和修改。当前复现范围是数学任务的 OPD：4B 学生 + 8B 教师；后续按具体目的开展工作。

## 2. Git 管理
Git 由 Agent 自主管理。根据不同工作目的自行创建、切换分支并提交改动，无需逐次确认；复现与不同方向的探索分别管理。

## 3. 历史记录
持续维护 `HISTORY.md`，按时间追加记录，保留已有历史，供后续回顾复盘。记录内容包括：

- 看过的数据、对比方式、要求分析的内容及结论。
- 代码修改、实验配置、检验结果，以及相关日志和产物的路径。

## 4. SwanLab 实验记录
本地环境已登录 SwanLab。所有实验，无论在本地还是远程运行，都必须上传到 SwanLab，记录实验配置、指标、结果和相关日志，并将实验链接写入 `HISTORY.md`。

## 5. 本机数学 TRD 暂定默认配置（2026-11-15）

后续会话进行本机数学 OPD/TRD 时，默认加载 [balanced 配置](preparation/pilot/profiles/balanced.env.sh)，以该文件为参数来源；`env.sh` 中旧的 batch=1 / 累积=64 不作为当前默认。根据新的实验目的调整时，在 `HISTORY.md` 记录差异。

- **硬件与并行**：2×RTX PRO 6000 Blackwell 96GB；单机 FSDP2，分片大小=2，SP=1。
- **模型与方法**：Qwen3-4B-Instruct-2507 学生 + Qwen3-8B 教师；默认 `forward_kl_y_r`，全词表前向 KL，`KL_LOSS_BACKEND=recompute`，chunk=512；温度=1、KL token clip=0。
- **批量与长度**：每 DP rank batch=2，梯度累积=32，全局有效 batch=128；单条训练序列上限=18,432，单卡动态微批 token 预算=32,768，超预算自动拆微批。该序列上限低于原 TRD 默认的34,816，会更早裁剪教师初始解答提示块。
- **GPU 优化与显存**：BF16 模型权重存储、FP32 LoRA 可训练参数；启用 FlashAttention 2、Liger、梯度检查点及 forward 后重分片；教师常驻 GPU，参数/优化器/策略 offload 均关闭；CUDA allocator fraction=0.92，以 NVML 整卡峰值约不超过90 GiB为目标。`USE_TORCH_COMPILE=false`（当前训练路径不计算 entropy，该开关没有已验证收益）。
- **其他训练参数**：LoRA rank=64 / alpha=128，学习率=5e-6；默认完整 DeepScaleR、1 epoch，保留生成输出上限16,384。试跑的少量样本、1,024输出上限不沿用为正式默认。
- **生成与评测**：rollout GPU memory utilization=0.85，max_num_seqs=16，max_num_batched_tokens=34,816；最终评测 AIME24/AIME25/HMMT25/BeyondAIME/AMO-Bench，pass@16，评测 TP=2；SwanLab 在线项目 `trd-math-opd`。

在仓库根目录启动（默认使用本地已准备的模型和数据）：

```bash
source preparation/env.sh
source preparation/pilot/profiles/balanced.env.sh
bash preparation/run_math_opd.sh forward_kl_y_r
```

只核对配置时，在最后一条命令末尾加 `--dry-run`。实测结果与适用边界见 [首轮报告](preparation/pilot/REPORT.zh-CN.md)；高容量 offload 备选不作为默认。
