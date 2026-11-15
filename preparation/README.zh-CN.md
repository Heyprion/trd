# 数学任务 OPD：4B 学生 + 8B 教师

本目录只准备本仓库数学 OPD 的环境、模型、数据和运行入口。

当前正式启动入口（自动加载 balanced 和冻结9,984题子集）：

```bash
bash preparation/run_train_eval.sh --dry-run
bash preparation/run_train_eval.sh --run-id math-trd-9984-v1
```

训练与评测分别记录到两个SwanLab实验。独立运行用 `run_train.sh` 和 `run_eval.sh --train-run <train/summary.json>`；模型导出会将LoRA合入HF权重后再评测。完整说明见[训练与评测脚本](training/README.zh-CN.md)。后文 `run_math_opd.sh` 是保留的上游通用入口。

当前数据工程已完成：可用33,236题，固定验证512题，训练池32,724题，首轮训练9,984题。完整过程、分层规则和文件索引见[数据工程完整记录](data_engineering/REPORT.zh-CN.md)；当前数据配置加载`preparation/data_engineering/splits.env.sh`。

2026-11-15 已完成双卡首轮性能试跑。下一次 TRD 运行建议显式加载
`preparation/pilot/profiles/balanced.env.sh`：FSDP2/2 卡、SP=1、每卡 batch=2、累积=32，
32K 动态 token 预算、单条上限 18,432，开启 Liger 与全词表 KL 重计算。
满预算压力测试整卡峰值约 88 GiB；详见 [性能报告](pilot/REPORT.zh-CN.md) 和
[试跑工具说明](pilot/README.zh-CN.md)。下文的 batch=1/累积=64 是最初准备配置，仍保留为未加载 profile 时的默认值。

本机准备已完成：两个模型共约 23 GiB 的权重已下载并通过 SHA256 校验；完整 DeepScaleR 共 40,315 条，五个评测集分别为 30 / 30 / 30 / 100 / 39 条。依赖检查、两卡 FlashAttention 前向/反向、NCCL、数学判分、实际 Ray/vLLM 学生生成，以及 4B 学生 + 8B 教师的两卡单步 TRD KL 更新均已通过。记录见 `readiness.json`、`validation.json` 和 `logs/`。正式完整训练尚未运行。

- 学生：`Qwen/Qwen3-4B-Instruct-2507`
- 教师：`Qwen/Qwen3-8B`
- 训练来源：`agentica-org/DeepScaleR-Preview-Dataset`，完整原始版本保留；当前训练范围按上述数据工程记录选择9,984题子集。
- 评测：AIME24、AIME25、HMMT25、BeyondAIME、AMO-Bench。AMO-Bench 沿用原脚本，过滤 `answer_type=description`。
- 支持原仓库全部五种数学 OPD 变体。`forward_kl_y_r` 是论文 TRD 方法；`forward_kl` 是普通 OPD 前向 KL 基线。

## 本机环境

Python 3.12；独立 `.venv` 继承宿主已有的 PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128，其他依赖写入 `.venv`。不要删除宿主 Conda 环境。

核心依赖固定为 vLLM 0.11.0、Transformers 4.57.1、Ray 2.49.2、FlashAttention 2.8.3。间接依赖版本记录在 `requirements-resolved.txt`，重建时也使用这些约束。FlashAttention wheel 对应 Python 3.12 / CUDA 12 / torch 2.8 / CXX11 ABI TRUE，不能换成原安装示例中的 ABI FALSE。

本机为 2 × RTX PRO 6000 Blackwell 96GB。原 wrapper 默认 8 GPU、每卡 batch=1、累积=16；本机配置采用 2 GPU、每卡 batch=1、累积=64，保持有效批量 128。LoRA rank=64 / alpha=128、学习率、生成温度、上下文长度、epoch 和评测 pass@16 沿用仓库默认。硬件并行度和累积次数变化仍可能导致数值差异；完整长序列训练的显存峰值须以实际运行验证为准。

## 使用

在仓库根目录执行：

```bash
source preparation/env.sh
source preparation/pilot/profiles/balanced.env.sh
source preparation/data_engineering/splits.env.sh
python preparation/check_ready.py --gpu
torchrun --standalone --nproc-per-node=2 preparation/check_nccl.py

# 查看参数，不生成、不训练
bash preparation/run_math_opd.sh forward_kl_y_r --dry-run

# TRD 方法
bash preparation/run_math_opd.sh forward_kl_y_r

# 普通 OPD 基线（独立实验）
bash preparation/run_math_opd.sh forward_kl
```

可选的运行链路检查：`bash preparation/smoke_rollout.sh` 只生成一条最多 32 token 的响应；`torchrun --standalone --nproc-per-node=2 preparation/smoke_train.py` 使用两条合成数学样本做一次 TRD 全词表 KL 更新，不保存 checkpoint。二者用于验证环境，不能作为复现实验结果。训练检查使用短序列并关闭 torch compile，完整上下文训练仍需单独运行。

其余变体名称：`forward_kl_clip`、`reverse_kl`、`reverse_kl_topk`。每次命令只运行一种变体，不会自动串行启动所有实验。默认训练后自动评测，W&B 使用离线模式。

`MODEL_PATH` / `TEACHER_MODEL_PATH` 指向本地下载，`TRAIN_DATA_PATH` 指向本地保存的 Hugging Face Dataset，评测位于 `data/eval/<name>/<name>_test.parquet`。`assets.json` 记录精确下载 revision 与文件大小。`readiness.json` 仅在检查全部通过后生成，`installed-freeze.txt` 记录实际安装版本。

## 在相同基础镜像重建

```bash
bash preparation/install.sh
source preparation/env.sh
python preparation/download_assets.py
bash preparation/prepare_data.sh
python preparation/check_ready.py --gpu
```

下载程序读取现有 `assets.json` 时复用已记录 revision。下载可断点续传；模型直接保存在 `models/`，不在 Hugging Face blob cache 再保存一份。网络需要镜像时可在启动下载前设置 `HF_ENDPOINT=https://hf-mirror.com`。大模型可用 `python preparation/download_assets.py --parallel-http` 启用分段续传并逐个校验权重的 SHA256。默认使用 HTTP 下载；启用 Xet 可设置 `HF_HUB_DISABLE_XET=0 HF_XET_HIGH_PERFORMANCE=1`。

本机 ModelScope 下载更快，可用下面的命令读取镜像权重。版本依据和 SHA256 仍取自 `assets.json` 记录的 Hugging Face revision；镜像内容不匹配时会报错，不能通过校验。

```bash
python preparation/download_assets.py --only models --parallel-http \
  --weight-url-template 'https://modelscope.cn/models/{repo_id}/resolve/master/{filename}'
```

模型、原始数据、环境、日志和本地缓存均不加入 Git。磁盘需额外容纳 rollout、训练 checkpoint 和导出模型；两个基础模型本身约 23 GiB，完整实验会继续增加空间占用。

本机数据盘 50GB，系统盘 30GB。原训练器会保存完整 FSDP 模型状态，且学生参数默认以 FP32 保存，因此不能只按 LoRA adapter 大小预留空间。启动脚本把 FSDP checkpoint 放在 `/root/trd-checkpoints`，并把该目录下的 `hf_merged` 链接到数据盘 `checkpoints_merged/`。训练器及评测仍使用原来的 `.../epoch1/hf_merged` 路径。启动前分别检查至少 20 GiB / 18 GiB 的空闲空间；同盘时要求合计 38 GiB。这是当前机器运行单个实验的安排，多个变体的模型一起保留需要扩容，或自行归档已有实验。换大磁盘时设置 `MODEL_SAVE_DIR`、`HF_EXPORT_ROOT` 即可。
