# 固定验证集 PPL / KLD 评分

当前输入是已冻结的 `data/processed/math-v1/splits-v1/validation/dataset`：512题，易69／中331／难112，原始参考解答共234,375 tokens。不会重新划分、改写参考解答或运行数学benchmark答案生成。

## 启动

在仓库根目录直接执行；启动器自动选择本仓库 `.venv/bin/python`，无需先加载训练环境。

```bash
# 只检查512题、哈希、训练池隔离、token数量和本地模型配置，不调用CUDA/SwanLab
bash preparation/run_ppl_kld.sh --dry-run

# 正式计算蒸馏前基线：原始4B学生 + 固定8B教师，默认完整512题
bash preparation/run_ppl_kld.sh
```

也可以从任意目录运行绝对路径：

```bash
bash /root/autodl-tmp/trd/preparation/run_ppl_kld.sh
```

如果想先做32题速度校准，显式使用下面的独立运行；它也在线上传SwanLab，但不会覆盖完整基线指针。

```bash
bash preparation/run_ppl_kld.sh --label calibration --max-samples 32
```

正式训练后，传入HF合并模型目录，或者含 `adapter_config.json` / `adapter_model.safetensors` 的LoRA目录；其他参数保持一致。

```bash
bash preparation/run_ppl_kld.sh \
  --label distilled \
  --student-model /实际路径/蒸馏后模型或LoRA目录 \
  --compare-to preparation/logs/ppl-kld/latest-baseline.json
```

LoRA默认基于 `models/Qwen3-4B-Instruct-2507` 加载；需要更换原始学生路径时传 `--base-model /实际路径/原始4B`。FSDP分片目录不能直接代替HF模型或LoRA目录，须先使用训练流程导出的模型。脚本不会下载权重。

路径及资源默认值集中在 [config.env.sh](config.env.sh)，支持同名 `PPL_KLD_*` 环境变量或CLI覆盖。`--help`可查看全部选项。例如单卡运行：

```bash
bash preparation/run_ppl_kld.sh --gpu-ids 0
```

## 本轮指标口径

- 教师、学生都看相同的 `problem + 空格 + instruction_following + 换行`，其格式与当前训练器的学生提示一致；解答仍是冻结记录中的原始 `solution`，分开编码后拼接，不做strip改写、不额外插入chat/thinking标签或EOS。
- 沿固定参考前缀做teacher-forcing，只统计response token，包括首token与末token；不统计题目、指令和padding。温度1、自然对数、全词表、不裁剪、无top-k。
- 输出学生/教师NLL与PPL、forward KL(T||S)、reverse KL(S||T)，KL单位为nats/token。全局先加总分子及有效token数再计算；不平均batch PPL。按易/中/难、领域和参考长度分桶。
- 模型BF16加载，LoRA保留PEFT的FP32 adapter推理转换；log-softmax/概率乘积为FP32，累积为FP64。直接取Qwen3 transformer隐藏状态，仅在response位置按chunk调用原LM head，避免生成整段 `[batch, length, vocab]` logits。没有梯度、优化器或KV cache。
- 这是**相同输入下的固定参考前缀KL**。TRD训练时教师额外看到初始解答及修订指令，训练日志KL与本脚本不同；不能把两者直接拼成一条曲线。本轮指标也不代表数学正确率或蒸馏后学生实时on-policy KL。
- 默认完整保留参考：长度超限、空解答、checksum变化、词表映射不一致、题组重叠及worker失败均报错，不会悄悄丢题或截断。

评分前会检查冻结manifest、validation/train_pool的parquet及ID文件SHA256，核对实际HF验证目录与parquet逐行一致，并检查验证与训练池的raw/sample/group ID零交集。更换验证集需提供另一份完整冻结split root。

## GPU、耗时与产物

默认两卡各一个“4B学生＋8B教师”完整副本，处理互不重叠的验证batch，无需FSDP/NCCL。batch最多8条、padded token预算8,192、全词表chunk512、FlashAttention 2、allocator fraction0.92。评分可以单卡运行；不改变训练balanced配置。峰值显存指标为PyTorch allocated/reserved，不是NVML整卡实测。

日志中会分别记录模型加载和纯评分耗时、有效token数量及每卡峰值。当前只做了CPU合成模型回归和真实512题dry-run，没有运行生产模型GPU评分；完整基线暂预留10–30分钟，实际以首轮或32题校准为准。正式运行还会读取模型文件计算SHA256，作为教师冻结和模型来源证据。

每次运行创建独立目录：

```text
preparation/logs/ppl-kld/<label>-<时间戳>-<pid>/
  config.json          # 输入、评分口径、模型文件SHA256、代码hash、软件版本
  per-sample.jsonl     # 每题NLL/KL分子、有效token数、前缀hash、分层；不含整段logits
  summary.json         # 全局/分桶统计、运行时间；可选前后配对差值和95%区间
  REPORT.md            # 可读报告
  evaluation.log       # 总日志
  gpu-*.log            # 逐GPU日志
  worker-*.json        # 逐GPU速度与显存
  swanlab.json         # 在线运行链接
```

完整且成功的 `baseline` 运行会更新 `preparation/logs/ppl-kld/latest-baseline.json`；校准、失败和其他label不会更新它。每轮运行自动上传SwanLab `trd-math-opd`并将链接、参数入口、结果或失败原因追加到 `HISTORY.md`。SwanLab失败时不回退离线运行。逐题数据、tokenized任务和日志只保存在被Git忽略的本地目录；在线上传汇总、配置和日志，不上传权重或题目文本。

`--compare-to`会校验基线结果SHA256、相同题目/前缀/token数/评分口径和固定教师权重，拒绝不一致的比较；输出current−baseline的变化及按题目配对bootstrap 95%区间（默认1,000次）。如要固定某次基线，直接传那次运行的 `summary.json`，而不是可被后续完整基线更新的latest指针。

## 开发校验

`test_scoring.py`使用CPU随机初始化的单层微型Qwen3和临时合成数据，检查与普通HF causal-LM loss及FP64全词表KL参考的一致性、BF16、padding/chunk不变性、LoRA加载、首末token、聚合、配对区间及污染/文件篡改拒绝。它不读取生产权重或给真实验证题打分。开发验证按Agent约定在线记录，运行链接见根目录HISTORY。
