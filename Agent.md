# Agent 工作约定

当前修复对照入口为 `bash preparation/run_repair_train_eval.sh`，仅训练用 `run_repair_train.sh`，可加 `--dry-run`。复用已完成首轮9,984题轨迹，从原始4B重新初始化LoRA，先保留全部题目及目标think，只修复训练聊天格式和结束监督。校验及实验状态见[修复说明](preparation/training/REPAIR.zh-CN.md)。旧复现入口与PPL/KLD协议保留，修复对照尚不代表已验证准确率收益。

## 1. 目的
复现他人的项目，并可能在其他项目上开展探索和修改。当前复现范围是数学任务的 OPD：4B 学生 + 8B 教师；后续按具体目的开展工作。

## 2. Git 管理
Git 由 Agent 自主管理。根据不同工作目的自行创建、切换分支并提交改动，无需逐次确认；复现与不同方向的探索分别管理。

远程仅同步代码、文档、配置及小型结果摘要。模型权重、数据集、虚拟环境、缓存、checkpoint、原始日志和本地 Git 备份不得上传；提交前检查文件清单和大小。

## 3. 历史记录
持续维护 `HISTORY.md`，按时间追加记录，保留已有历史，供后续回顾复盘。记录内容包括：

- 看过的数据、对比方式、要求分析的内容及结论。
- 代码修改、实验配置、检验结果，以及相关日志和产物的路径。

## 4. SwanLab 实验记录
本地环境已登录 SwanLab。所有实验，无论在本地还是远程运行，都必须上传到 SwanLab，记录实验配置、指标、结果和相关日志，并将实验链接写入 `HISTORY.md`。

- **实验平台**：数据处理、生成、训练、评测、性能试跑和数值校验等实验均经过 SwanLab 记录；失败或中断的实验同样保留状态、原因和日志。
- **项目复用**：启动前查看已有 project 和 `HISTORY.md`。主要目的和阶段相同，优先复用原 project；换日期、机器、模型、超参或重新运行不单独新建 project。阶段变化也先判断原 project 是否适用，只有主要目的或阶段显著变化且现有 project 不适合时才新建。当前数学 OPD/TRD 的数据、生成、训练及评测默认复用 `trd-math-opd`；硬件性能专项优先复用已有 `trd-pro6000-pilot`。
- **实验命名**：统一使用 `YYYYMMDD-阶段-目的[-关键区别][-rNN]`，必须包含日期、阶段和直白的目的。阶段统一使用“数据、生成、训练、评测、性能、数值校验”等明确词；只按需要补充样本量、方法或基线/蒸馏后等关键区别，同日同目的重跑加 `r02`、`r03`。完整超参和机器信息写入 config/description，不堆进名字，也不使用纯随机串或含义不清的临时名字。

命名示例：`20261115-数据-清洗分层`、`20261115-训练-TRD-9984题`、`20261115-评测-PPL和KL-基线`、`20261115-评测-五套测试-pass16`。

## 5. 本机数学 TRD 暂定默认配置（2026-11-15）

后续会话进行本机数学 OPD/TRD 时，默认加载 [balanced 配置](preparation/pilot/profiles/balanced.env.sh)，以该文件为参数来源；`env.sh` 中旧的 batch=1 / 累积=64 不作为当前默认。根据新的实验目的调整时，在 `HISTORY.md` 记录差异。

- **硬件与并行**：2×RTX PRO 6000 Blackwell 96GB；单机 FSDP2，分片大小=2，SP=1。
- **模型与方法**：Qwen3-4B-Instruct-2507 学生 + Qwen3-8B 教师；默认 `forward_kl_y_r`，全词表前向 KL，`KL_LOSS_BACKEND=recompute`，chunk=512；温度=1、KL token clip=0。
- **批量与长度**：每 DP rank batch=2，梯度累积=32，全局有效 batch=128；单条训练序列上限=18,432，单卡动态微批 token 预算=32,768，超预算自动拆微批。该序列上限低于原 TRD 默认的34,816，会更早裁剪教师初始解答提示块。
- **GPU 优化与显存**：BF16 模型权重存储、FP32 LoRA 可训练参数；启用 FlashAttention 2、Liger、梯度检查点及 forward 后重分片；教师常驻 GPU，参数/优化器/策略 offload 均关闭；CUDA allocator fraction=0.92，以 NVML 整卡峰值约不超过90 GiB为目标。`USE_TORCH_COMPILE=false`（当前训练路径不计算 entropy，该开关没有已验证收益）。
- **其他训练参数**：LoRA rank=64 / alpha=128，学习率=5e-6；1 epoch，当前数据范围遵循第6节的暂定划分，保留生成输出上限16,384。试跑的少量样本、1,024输出上限不沿用为正式默认。
- **生成与评测**：rollout GPU memory utilization=0.85，max_num_seqs=16，max_num_batched_tokens=34,816；按2026-11-16用户最新要求，当前最终评测仅 AIME24/AIME25，pass@16，评测 TP=2；其他三个测试集不默认执行。SwanLab 在线项目 `trd-math-opd`。

在仓库根目录启动（默认使用本地已准备的模型和数据）：

```bash
source preparation/env.sh
source preparation/pilot/profiles/balanced.env.sh
source preparation/data_engineering/splits.env.sh
bash preparation/run_math_opd.sh forward_kl_y_r
```

只核对配置时，在最后一条命令末尾加 `--dry-run`。实测结果与适用边界见 [首轮报告](preparation/pilot/REPORT.zh-CN.md)；高容量 offload 备选不作为默认。

## 6. 数据工程与暂定划分索引（2026-11-15）

原始数据、抽样观察、清洗/分层、训练—测试排重及当前划分统一遵循[数据工程完整记录](preparation/data_engineering/REPORT.zh-CN.md)。当前固定验证512题、训练池32,724题，首轮仅训练其中分层抽取的9,984题（有效batch128，共78个名义更新/epoch）；验证题及同题组不得参与训练或训练生成。加载`preparation/data_engineering/splits.env.sh`选择当前子集和独立生成缓存；后续扩充保留验证清单，沿固定训练池顺序取更长前缀。原始全量数据保留作复现版本，不是当前默认训练输入。

## 7. 固定验证集 PPL/KLD 入口（2026-11-15）

在根目录执行`bash preparation/run_ppl_kld.sh`计算蒸馏前基线，默认固定512题、原始4B学生与冻结8B教师、双卡分片评分；`--dry-run`只核对输入。训练后用`--label distilled --student-model <HF模型或LoRA目录> --compare-to preparation/logs/ppl-kld/latest-baseline.json`进行同口径比较。指标为相同题目/参考前缀上的response PPL与全词表forward/reverse KL，区别于教师额外看初始解答的TRD训练KL；详见[评分说明](preparation/ppl_kld/README.zh-CN.md)。实际运行自动上传SwanLab并追加HISTORY；准备脚本不代表已经完成基线评分。

截至2026-11-16，完整512题基线及首轮9,984题训练已完成；本轮蒸馏后评分直接执行`bash preparation/run_ppl_kld_distilled.sh`，固定读取已完成训练记录中的合并模型与既有基线summary，可加`--dry-run`核对。该入口只准备评分，不表示蒸馏后PPL/KLD已算完。

## 8. 正式训练与独立评测入口（2026-11-15）

当前推荐一键入口为 `bash preparation/run_train_eval.sh`，自动加载 balanced 与冻结9,984题子集，训练成功并完成LoRA合并后再执行AIME24/AIME25的pass@16评测（2026-11-16缩减范围）；其他测试需显式设置MATH_EVAL_DATASETS。加 `--dry-run` 仅核对数据与配置。单独训练用 `bash preparation/run_train.sh`；单独评测用 `bash preparation/run_eval.sh --train-run <train/summary.json>` 或 `--model <已合并HF目录>`。训练和评测分别创建SwanLab实验，并追加HISTORY；详细命令、日志及检查说明见[训练评测入口说明](preparation/training/README.zh-CN.md)。固定512题PPL/KLD继续使用第7节的独立入口。

蒸馏前学生的AIME24/AIME25同口径基线用 `bash preparation/run_base_eval.sh`（2026-11-16新增）；固定原始4B模型、不加载LoRA，读取已完成蒸馏评测的参数与测试集哈希，并核对60题分词输入。base缓存和成绩独立保存，结束后自动生成对比表；支持 `--reference-eval`更换参考summary，`--dry-run`仅检查。该benchmark基线与第7节固定参考PPL/KLD基线是两个独立实验。
