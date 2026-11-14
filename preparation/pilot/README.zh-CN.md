# 2×RTX PRO 6000 数学 OPD 性能试跑

本目录用于少量样本测试真实 FSDP2 训练链路，不做收敛评测。结果与建议见 `REPORT.zh-CN.md`；完整指标索引见 `results.json`。

运行前：

```bash
source preparation/env.sh
python preparation/pilot/prepare_stress.py
```

固定长度压力输入来自 DeepScaleR 前 32 个真实问题/解答，重复解答文本以达到目标长度，并由数据集逻辑裁剪到 max_length。当前生成器为每种长度准备 128 行（32 个独立来源），初期长序列试验使用 32 行。它们只测算力、显存和稳定性，不能代表 on-policy 分布或正确率。真实 student rollout → teacher refine → KL 更新检查由 `rollout_probe.py` 和真实 parquet 训练完成。

例：

```bash
python preparation/pilot/run_probe.py \
  --name example-18k --length 18432 --budget 36864 --batch 2 --steps 8 \
  --options '{"kl_loss_backend":"recompute","model_storage_dtype":"bf16","use_liger":true,"teacher_cpu_offload":true,"cuda_memory_fraction":0.92,"accum":4,"refine":true}'
```

`--data` 可以指定真实生成 parquet。`batch` 是每个 DP rank 的外层 batch，`budget` 是每 rank 单个动态微批的 token 上限；外层 batch 大于预算时会拆分，教师输出仍会保留整个外层 batch，不能单靠缩小 token budget 保证任意外层 batch 都安全。`length` 是每条输入的上限。`steps` 是外层微批数，不是有效全局 batch=128 的优化器步数。性能记录排除第一次预热，按较慢 rank 的时间和不重复的 DP token 数计算吞吐。

`run_probe.py` 会在线初始化 SwanLab，再运行两个 torchrun rank；每 0.1 秒采样 NVML。所有配置、失败、训练日志、汇总指标上传 SwanLab，链接自动追加到 `HISTORY.md`。同名目录已有数据时自动追加时间，防止覆盖旧结果。工作区源码快照和 diff 保存在每次运行的 source 目录，数据 SHA256 自后期试验起也写入配置。

每次结果：`preparation/logs/pilot/<name>/`：

- `case.json`、`config-rank*.json`：命令选项和实际 trainer 配置。
- `metrics-rank*.json`、`dtypes-rank*.json`：每 rank 实际长度、KL、梯度范数、耗时、PyTorch 峰值和参数存储精度。
- `telemetry.json`、`summary.json`：NVML 整卡显存和利用率、成败、SwanLab 链接。
- `train.log`、`error-rank*.json`：完整 stdout 和异常。

数值检查：

```bash
python preparation/pilot/check_recompute.py
CHECK_DEVICE=cuda python preparation/pilot/check_autocast.py
```

前者检查掩码、温度和裁剪；后者在完整 151,936 词表和 BF16 autocast 下，对照稳定 FP64 KL 与梯度。不要使用受 autocast 影响的原 einsum 作为高精度参考。原实现保留为 `KL_LOSS_BACKEND=autograd`，新实现为 `recompute`，支持正向和反向全词表 KL，不支持 top-k / JSD，默认不改变其他变体。

```bash
python preparation/pilot/summarize.py
```

重新生成结果索引。`run_*matrix.py` 是本轮试验的候选顺序；重跑会使用当前代码，历史逐行 KL v1 的实现与当前 packed v2 有差异，应以历史说明和 source 快照为准。

训练入口新增可选环境变量：`KL_LOSS_BACKEND`、`MODEL_STORAGE_DTYPE`、`USE_LIGER`、`TEACHER_CPU_OFFLOAD`、`RESHARD_AFTER_FORWARD`、`GRADIENT_CHECKPOINTING`、`CUDA_MEMORY_FRACTION`。普通 KLTrainer 默认在线 SwanLab（项目 `trd-math-opd`），probe 则由父进程统一记录（项目 `trd-pro6000-pilot`），避免每个 rank 创建独立实验。
