# 数学 OPD：2×RTX PRO 6000 首轮试跑

推荐下一次数学 OPD/TRD 运行从 `profiles/balanced.env.sh` 起步：FSDP2 分片大小 2、SP=1，每 DP rank batch=2、梯度累积=32（全局有效 batch=128），动态微批预算 32,768 tokens，单条序列上限采用本轮试跑的 18,432。冻结模型权重以 BF16 保存，学生 LoRA 参数保持 FP32；开启 Liger 与梯度检查点，使用分块重计算全词表 KL、chunk=512，教师常驻 GPU，forward 后重分片，CUDA 分配器份额 0.92。LoRA r64/alpha128、学习率 5e-6、温度 1、KL clip=0 保持原方法设置。

运行方式（本轮仅执行了少量样本试跑，没有启动完整数据训练）：

```bash
source preparation/env.sh
source preparation/pilot/profiles/balanced.env.sh
bash preparation/run_math_opd.sh forward_kl_y_r --dry-run
# 正式运行入口：去掉 --dry-run；完整训练后会按原入口评测。
```

32K 动态预算下，两个完整 18K 样本会拆成两个微批；每卡外层实际总 token 数仍可达 36,864。此压力输入平均约 10.33s/外层微批，峰值 76.72 GiB，对照原 batch=1 的两个 18K 微批需约 13.35s，吞吐提升约 29%。另用每卡两条 16K 输入实际填满 32K 预算，带梯度累积验证，峰值 87.93 GiB。NVML 稳态 GPU 利用率约 92%～98%，取决于微批拆分和优化器初始化。

更强调长序列微批容量时，可使用 `profiles/high_memory.env.sh`：预算 36,864，教师 CPU offload；两个完整 18K 样本连续 8 个微批、两次累积更新通过，整卡峰值 88.44 GiB、吞吐约 6.28K response tokens/s，慢于 balanced 的约 7.08K。显存占用更高并不意味着吞吐更高。

所有显存数字为 NVML 整卡峰值（包括 PyTorch 保留缓存及 NCCL/驱动等），以 MiB/1024 换算 GiB；不把仅看 allocated 的 84 GiB 等同于整卡 84 GiB。0.92 是 PyTorch 分配器预算，不是整卡绝对硬限制，选择依据是实测 NVML 峰值；最终推荐均需以实际数据继续观察。

## 测量与比较边界

长度范围更正：原 `forward_kl_y_r.sh` 的 MAX_LENGTH 默认是 34,816；本轮 balanced profile 显式改为 18,432，会更早裁剪教师的 initial-response 提示块。因此本轮并非原始最大上下文长度的完整复现；原 34,816 长度的显存和吞吐尚未覆盖。

- 固定长度压力输入来自真实数学文本，必要时重复到指定长度，不能用于准确率/收敛结论。每次实际长度、梯度范数、KL、耗时和两类 PyTorch 峰值均有记录。压力输入只有 32 个独立来源，后期短样本用 128 行重复索引支持 batch=16 的多步测试。
- 吞吐按较慢 rank 的时间、两卡不重复的 DP token 数计算；丢弃第一轮预热。大多数候选测 3 个微批，推荐高容量配置测 8 个；这是首轮候选筛选，不能视为完整长跑稳定性或收敛保证。
- SP=2 单次耗时约 3.87s，但两卡只处理一份序列，吞吐约 4.74K tokens/s；SP=1 两卡处理独立数据约 7.1K，故保留 SP=1。不能只比较单次壁钟时间。
- FP32 教师常驻 + 大批量触发 OOM；BF16 常驻直接 batch=2/18K 可运行但 NVML 约 95.5 GiB，余量不足。大批量 no-reshard 和长序列关闭 checkpointing 均 OOM。失败日志、配置和 SwanLab 均保留。
- use_torch_compile 在当前引擎只用于 entropy；本轮关闭 entropy，因此该开关不产生已证实的训练加速。

## 数值实现与优化依据

参考 [Liger Kernel 论文](https://arxiv.org/abs/2410.10989) 的分块与减少中间量思路，接入已安装的 Liger Qwen3 RMSNorm/RoPE/SwiGLU 内核，另实现全词表 KL 的重计算反向。它保存 BF16 logits，在有限 FP32 工作区重新计算 softmax，整批共享一个反向节点，避免逐行 nested unbind 的大梯度散射。当前仍物化完整 LM head logits，因此没有宣称达到 fused-linear 蒸馏内核的内存上限。

24 组掩码/温度/裁剪数值与梯度检查通过。随后在 CUDA 和完整 151,936 词表、BF16 autocast 环境对照稳定 FP64 参考：重计算版最大前向误差 <1.2e-6，近似相同分布的 forward KL 梯度相对误差约 0.012%；原 einsum 实现约 8.03%。原实现的 FP32 输入 einsum 会被 autocast 降精度，且长归约也不适合作为高精度参考。新旧目标同为全词表 KL，但不承诺旧 BF16 实现逐位一致，长期训练质量需另行评测。

数值检验：[CUDA 全词表/梯度检查](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/3ro3hcoy)，[温度/裁剪等检查](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/ac2k0sp6)。普通自定义反向只支持一阶梯度，足够当前 LoRA 优化器；不支持 top-k/JSD，其他变体保留原 autograd 后端。

额外修复了实际 train_epoch 循环末尾保留上一批 teacher logits 的引用，避免下一次教师前向与旧批量同时占显存。FSDP2 内存/通信选项遵循 [PyTorch 2.8 文档](https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html)，保留旧默认，仅在显式 profile 中改变选项。普通训练现默认记录在线 SwanLab，性能 probe 由父进程统一记录。

## 真实数据链路与完整结果

最终真实生成/更新记录与各候选结果表在本节下方。所有运行的目录和链接也逐项追加在根目录 `HISTORY.md`。原始日志不加入 Git，保存在 `preparation/logs/pilot/`；本目录保留工具、环境 freeze、配置与结果索引。

真实链路：[8 题学生生成与教师改写](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/2enpetf3)；生成长度仅为试跑设置上限 1,024，改写实际为 1,003～1,024 tokens，冷启动总耗时 296.9s（包含两次 Ray/vLLM 启动、编译和生成），NVML 峰值 79.74 GiB。此处没有验证原 16K 输出长度的完整 rollout 负载。

[真实样本 FSDP2 更新](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/ddcwfrbe)与[原生 train_epoch 日志](https://swanlab.cn/@WangZIyu/trd-math-opd/runs/fxue8aiu)均通过；每 rank 执行 2 个 probe 更新和 2 个原生更新，共 4 次优化器更新；KL 与梯度均有限。没有保存 checkpoint、没有运行完整集准确率评测。

短样本补测：2K 序列每卡 batch=16、32K 预算、分配器份额 0.92，整卡峰值 88.06 GiB，吞吐约 8.97K response tokens/s，为原 batch=1 的约 3.13 倍。这一 batch 仅适用于已经限制为短样本的压力测试，不能直接作为完整长短混合数据的外层 batch。

推荐有效全局 batch 128 是 2 GPU × 每卡 batch 2 × 累积 32；本轮长序列累积检查使用 2/4 次累积以缩短试跑，并验证了梯度及 Adam 状态常驻后的显存；没有完成全局 batch=128 的长程训练。

| 试验（SwanLab） | 结果 | 单卡微批预算 | 每卡 batch | NVML 峰值 GiB | 外层微批秒 | response tokens/s |
|---|---|---:|---:|---:|---:|---:|
| [baseline-18k](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/a906hp9e) | 通过 | 18432 | 1 | 59.79 | 6.676 | 5496 |
| [baseline-2k](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/8dnul6rc) | 通过 | 2048 | 1 | 19.53 | 1.366 | 2869 |
| [bf16-18k-b2](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/7dlotxzz) | 通过 | 36864 | 2 | 95.47 | 13.088 | 5607 |
| [bf16-18k-b2-liger](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/5ptdslru) | 通过 | 36864 | 2 | 95.31 | 11.487 | 6388 |
| [bf16-18k-b2-noreshard](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/74b8x163) | OOM | 36864 | 2 | 95.49 | — | — |
| [bf16-18k-no-gc](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/5q7qz65b) | OOM | 18432 | 1 | 95.32 | — | — |
| [bf16-2k-b8](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/wiuz66io) | 通过 | 16384 | 8 | 56.42 | 4.128 | 7631 |
| [capped-offload-18k-b2-acc4](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/hdw9dx5x) | 通过 | 36864 | 2 | 88.44 | 11.656 | 6279 |
| [capped-resident-16k-b2](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/7dqnisz5) | 通过 | 32768 | 2 | 87.93 | 9.313 | 6986 |
| [capped-resident-2k-b16](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/tqvk3qz1) | 通过 | 32768 | 16 | 88.06 | 7.001 | 8969 |
| [dynamic32k-resident-18k-b2](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/vbcosoyz) | 通过 | 32768 | 2 | 76.72 | 10.327 | 7080 |
| [offload-liger-18k-b2](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/alb15fup) | 通过 | 36864 | 2 | 95.51 | 10.306 | 7121 |
| [real-opd-trd-update](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/ddcwfrbe) | 通过 | 32768 | 2 | 26.21 | 1.138 | 3599 |
| [recompute-18k](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/50oqfkd1) | 通过 | 18432 | 1 | 58.62 | 6.035 | 6079 |
| [resident-18k](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/drsrf2ju) | 通过 | 18432 | 1 | 73.88 | 5.992 | 6123 |
| [resident-18k-b2](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/vzqyy8e1) | OOM | 36864 | 2 | 95.21 | — | — |
| [resident-18k-b3](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/4bol3f34) | OOM | 55296 | 3 | 95.57 | — | — |
| [resident-liger-16k-b2](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/1gjsnvpg) | 通过 | 32768 | 2 | 93.86 | 8.778 | 7427 |
| [resident-liger-18k-b1](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/ieert45i) | 通过 | 18432 | 1 | 61.09 | 5.162 | 7107 |
| [resident-liger-2k-b16](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/mumtgki0) | 通过 | 32768 | 16 | 93.86 | 6.693 | 9380 |
| [sp2-resident-18k-b1](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/wia4p2lb) | 通过 | 18432 | 1 | 48.85 | 3.873 | 4742 |

SwanLab [首轮结果汇总与推荐配置](https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/u6ewxwcl)。
