# 9,984 条数学 TRD：训练、评测与串行入口

聊天格式修复及复用既有轨迹的入口见[修复说明](REPAIR.zh-CN.md)，一键命令为 `bash preparation/run_repair_train_eval.sh`；该对照从原始4B重新训练，跳过两轮轨迹生成。

首轮实际结果及退化排查见 [AIME 蒸馏前后分析（2026-11-16）](reports/20261116-aime-regression.zh-CN.md)：两套平均单次正确率下降，PPL/KL 改善未转化为独立解题收益；文档区分已核实的输入/监督差异与尚待控制变量验证的原因。

仓库根目录直接运行，无需手动激活环境：

```bash
# 先检查冻结数据、配置和模型交接计划；不启动 GPU 或 SwanLab 实验
bash preparation/run_train_eval.sh --dry-run

# 正式运行：学生生成 → 教师改写 → FSDP2 训练 → 合并 LoRA → 独立评测
bash preparation/run_train_eval.sh --run-id math-trd-9984-v1
```

只训练或单独评测：

```bash
bash preparation/run_train.sh --run-id math-trd-9984-v1

# 读取这次成功训练的准确模型路径，不依赖 latest 或目录排序
bash preparation/run_eval.sh --run-id math-trd-9984-v1-test \
  --train-run preparation/logs/math-runs/math-trd-9984-v1/train/summary.json

# 也可明确传入已经合并 LoRA 的 HF 模型目录
bash preparation/run_eval.sh --model /absolute/path/to/epoch1/hf_merged --dry-run

# 蒸馏前4B学生：与已完成的AIME24/AIME25评测同口径，独立记录
bash preparation/run_base_eval.sh
```

三个入口均支持从任意目录以绝对脚本路径执行。`--run-id` 可省略，默认使用时间戳及进程号；已有运行目录或训练产物不会被覆盖。评测失败后可以新 run-id 加 `--train-run` 单独重跑评测，无需重新训练。运行期间请保持终端连接，或由用户的 tmux/任务管理器托管；脚本本身在前台执行。

`run_base_eval.sh`固定加载原始 `models/Qwen3-4B-Instruct-2507`，只测AIME24、AIME25，不加载LoRA、不训练。默认参考已完成蒸馏评测 `preparation/logs/math-runs/20261115-185850-279780-53607/eval/summary.json`，从中固定pass@16、温度0.6/top_p0.95、TP2、提示4096/输出38912/上下文43008、显存比例0.85、max_num_seqs16及batched tokens43008。数据行数与SHA256必须相同；另核对两个模型的generation defaults、词表/特殊token及全部60题的chat-template编码输入一致。沿用原生成器与判分器，单副本的默认seed0保持不变。

base脚本支持 `--dry-run`（仅检查，未启动GPU或SwanLab）、`--run-id`、`--output-root`、`--reference-eval <另一份完成的蒸馏评测summary.json>`；也可用 `BASE_EVAL_REFERENCE`设置参考路径。默认独立输出 `preparation/logs/math-base-eval/<run-id>/eval/`，以 `<run-id>_base`记录成绩，使用全新的生成缓存。SwanLab复用 `trd-math-opd`，命名为 `YYYYMMDD-评测-AIME24和25-pass16-基线`，同日重跑加r02等。结束后保存results.json/csv、summary、comparison.json/csv；比较列为base、蒸馏后与“蒸馏后减base”的百分点差值，同时上传SwanLab并追加HISTORY。只生成脚本/通过dry-run不代表已经得到base准确率。

## 数据及配置

自动按顺序加载 `env.sh`、`pilot/profiles/balanced.env.sh`、`data_engineering/splits.env.sh` 及原 `forward_kl_y_r.sh` 的算法默认值。训练固定为 `train_9984/dataset` 的全部9,984条，验证512条不参与生成或训练。启动前检查冻结 manifest、parquet/ID文件 SHA256、HF与parquet逐行相等，以及 raw_id/sample_id/duplicate_group_id 在训练/验证间无交集。拒绝原始全集、`MAX_SAMPLES`、外部 `DATA_PATH` 和 `CORRECTED_RESPONSES_PATH` 等绕过配置。

默认每卡 batch=2、累积32、有效batch128，1 epoch/78个名义更新；FSDP2/2卡、SP=1；全词表 forward KL、recompute/chunk512、BF16冻结权重及FP32 LoRA、Liger、梯度检查点、18,432序列上限与32,768动态预算。训练生成上限16,384。正式入口会校验生成结果恰好包含这9,984题且身份/顺序一致；如果上游长度过滤或不完整缓存导致丢题，将在训练前报错，记录日志并停止，不静默改小数据集。

按2026-11-16用户要求，默认只评测 AIME24/AIME25（各30题），每题16次采样、温度0.6、top_p0.95、TP=2；报告 pass@16，同时记录 avg@16 与首8次的 pass@8。HMMT25/BeyondAIME/AMO-Bench数据仍保留，但不默认运行；需要其他组合时显式设置 `MATH_EVAL_DATASETS=aime24,aime25,hmmt25`。保持原 benchmark 的提示预算4,096、输出上限38,912（这与训练生成的16,384分开）；评测 vLLM 显存比例0.85、max_num_seqs16、batched tokens43,008。通过 `EVAL_RESPONSE_LENGTH` 等环境变量修改会进入评测配置记录。自定义训练资源配置可用 `MATH_TRAIN_PROFILE=/absolute/path/profile.env.sh`，会替代 balanced；配置差异与参数快照随实验留档。

FSDP 导出时显式 `--merge-lora`，将 LoRA delta 折入最终HF权重后保存，并生成 `lora_merge.json`；原adapter仍保存在模型目录的 `lora_adapter/`。评测拒绝存在adapter却没有合并完成标记的旧导出，避免只测基础模型。检查全部权重分片存在；训练/导出失败时不启动评测。原上游入口保留旧行为，新三个脚本统一使用修复后的合并路径。

## 实验记录与产物

同一总运行在 SwanLab 项目 `trd-math-opd` 下创建两个独立实验，名称为 `<run-id>-train` 与 `<run-id>-eval`，以run-id分组。训练实验覆盖生成、更新、导出，转发rank0的原生训练指标；评测实验记录所选测试集的结果和来源训练链接。在线初始化失败会终止运行。控制台由父进程转发并由SwanLab采集；成功与失败均写入 `HISTORY.md`。总脚本会先等待训练进程及模型导出完成，再释放训练资源并启动独立评测。

运行中的评测可通过其结果目录下 `evaluation-control.json` 在测试集边界缩减范围。旧版已启动进程不能热更新已加载的循环时，用PID与结果路径匹配的判分后退出钩子，先保留最后一套的响应和分数、清理Ray，再结束旧循环；`finish_reduced_eval.py` 等旧控制器退出后仅对所选集合汇总，并恢复同一个SwanLab评测记录。原控制器可能留下预期的退出异常；最终summary会注明用户变更范围，并保留旧summary作审计。此操作不暂停正在生成的测试集，不修改测试数据，也不重新加载模型做评测。

本地配置、SwanLab链接、状态、异常及结果位于：

```text
preparation/logs/math-runs/<run-id>/train/{summary.json,console.log,metrics.jsonl,swanlab/}
preparation/logs/math-runs/<run-id>/eval/{summary.json,console.log,results.json,results.csv,generations/,swanlab/}
```

训练 checkpoint、合并模型和生成缓存仍沿用 `MODEL_SAVE_DIR`（默认 `/root/trd-checkpoints`）、`HF_EXPORT_ROOT`（默认 `checkpoints_merged/`）与 `gen_results/math-v1-splits-v1/train_9984/`，具体路径在summary内。保持原来的分盘空间检查：checkpoint盘至少20GiB，导出盘18GiB，同盘38GiB。模型、数据、原始日志和缓存均不进入Git。脚本以文件锁防止这些新入口同时占用同一套GPU。

固定512题参考PPL/KLD仍由已准备的专用脚本单独运行，属于另外的验证实验；已有baseline链接见 `HISTORY.md`。如需训练后同口径比较：

```bash
bash preparation/run_ppl_kld.sh --label distilled \
  --student-model /absolute/path/to/epoch1/hf_merged \
  --compare-to preparation/logs/ppl-kld/latest-baseline.json
```

本次准备只执行数据/配置检查和CPU合成测试，没有启动正式9,984题生成、训练或五个benchmark评测。
