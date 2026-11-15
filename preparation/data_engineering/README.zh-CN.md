# 一键数学数据清洗与分层

两个阶段：`clean.py`全自动清洗，`stratify.py`给清洗通过的题标数学领域和易/中/难。`run.sh`默认连续执行两阶段，两个8B GPU副本全程复用。原始数据只读，不修改训练默认配置，不进行完整解题或训练。

当前清洗、标签和实际划分均已完成，完整过程与当前数据使用范围以[数据工程完整记录](REPORT.zh-CN.md)为准：验证512题、训练池32,724题、首轮训练9,984题。已冻结的同一输出目录不要重复清洗或重新抽验证集。

在本仓库根目录直接运行：

```bash
bash preparation/data_engineering/run.sh
```

默认使用现有`.venv`、`data/train/DeepScaleR`、`data/eval`及`models/Qwen3-8B`，双卡0/1、CPU16 workers、总预算3600秒。必须已有可用的SwanLab登录，实验强制在线记录到`trd-math-opd`并追加`HISTORY.md`。

## 放到另一台机器或目录

复制整个`data_engineering`目录或解压脚本包，保留所有Python文件。代码包不包含模型、数据、环境和日志。

准备Python3.12及兼容的CUDA/vLLM环境。已有本仓库环境无需安装；新环境可按`requirements.txt`安装依赖，并运行`swanlab login`登录。安装依赖、准备模型/数据属于首次环境准备，不计入批处理的一小时预算。

修改`config.env.sh`里的路径，或直接用环境变量：

```bash
INPUT_PATH=/work/data/DeepScaleR \
EVAL_DIR=/work/data/eval \
LABEL_MODEL=/work/models/Qwen3-8B \
OUTPUT_DIR=/work/results/math-v1 \
PYTHON_BIN=/work/venv/bin/python \
bash /work/data_engineering/run.sh
```

输入支持Hugging Face `save_to_disk`目录、JSON数组、JSONL、parquet，至少包含`problem`和`answer`，`solution`可以为空。评测目录需要准备实际使用的五套测试parquet，用于训练排重；目录缺失或没有测试题会立即报错。

原始Omni-MATH标签来源默认读取`OMNI_PATH`，不存在时自动下载约7.5MB的固定revision并校验SHA256，不下载模型。网络失败则记录日志，剩余题使用模型标签。`DOWNLOAD_OMNI=false`可以关闭下载。模型必须在本地，避免运行中意外下载大模型。

如需单独配置文件，设置`DATA_ENGINEERING_CONFIG=/path/custom.env.sh`；变量名见默认配置。`GPU_IDS`使用物理卡号或GPU UUID，默认继承`CUDA_VISIBLE_DEVICES`，未设置时为`0,1`。

## 独立执行两个阶段

同一套配置、同一个`OUTPUT_DIR`：

```bash
bash preparation/data_engineering/run.sh clean
bash preparation/data_engineering/run.sh label
```

也可直接执行`python clean.py ...`和`python stratify.py ...`，两者都支持`--help`。独立执行时默认各分配1800秒、各加载一次模型；一键执行总预算3600秒且只加载一次。分层只读取`OUTPUT_DIR/clean/`，不会重新判断质量，不需要保留原始输入路径。

常用检查和运行方式：

```bash
# 只检查路径和打印实际参数，不使用GPU或创建SwanLab实验。
bash preparation/data_engineering/run.sh --dry-run

# 少量真实样本验证；使用单独目录，不覆盖正式输出。
bash preparation/data_engineering/run.sh --max-samples 32 \
  --output preparation/logs/data-engineering/my-smoke --total-seconds 600 --export-reserve 60

# 改用其他GPU/CPU并行参数。
bash preparation/data_engineering/run.sh --gpu-ids 0,1 --workers 32 --max-num-seqs 128

# 复用已完成的清洗阶段，继续分层；输入、模型与规则指纹必须匹配。
bash preparation/data_engineering/run.sh --resume
```

已完成的输出目录默认拒绝覆盖。`--resume`会验证输入内容、测试集、来源、代码、模型配置及权重文件元数据后复用完成阶段；不会复用中断阶段中的部分GPU批次。想重新执行或更换规则时指定新的`OUTPUT_DIR`。同一输出目录有进程锁，禁止同时写入。

## 决策和标签

- 每条输入最后只有`usable`或`excluded`。缺标准答案、缺关键图片/条件、解答片段、题干泄露答案及无法自动闭环的题排除；空参考解答保留。
- 精确重复留一条；同题答案冲突或原始标签冲突整组排除。CPU召回近重复，GPU比较候选；最终测试同题从训练侧排除。
- 原始可靠标签优先使用；原Omni分数≤3映射easy、>3且≤5映射medium、>5映射hard。
- 无原标签的可用题由8B非思考模式自动标注；输出使用简短英文枚举，例如`geometry,medium`，最长8 tokens。领域只有algebra、geometry、number_theory、combinatorics_probability、calculus、other；难度只有easy、medium、hard。
- 标注格式错误或推理错误最多短重试一次，再失败排除。内容排除与技术超时/失败分开统计。小模型自动筛选与粗分层不提供数学正确性证明。

每卡调度上限默认128；数据量足够时在同一常驻模型上比较有效batch64/128，记录实测请求吞吐后选择。CPU处理与GPU加载并行。为缩短冷启动默认`ENFORCE_EAGER=true`；可以显式改为false，但不沿用此前训练吞吐作为标注速度。

默认内容处理最晚在30分钟的清洗预算内停止发新请求，全部GPU工作在总预算减去3分钟导出缓冲时停止，进行中的短批次最多额外等待60秒。未完成记录按技术原因排除并计数；父进程另有总时限。60分钟是运行上限设计，SwanLab网络异常、硬件故障等会以失败退出，不能保证全量样本一定完成。结果中的技术排除数量会直接显示覆盖损失。

## 输出位置

```text
OUTPUT_DIR/
  clean/
    usable.parquet          # 第一阶段通过的题，附可靠原始标签信息
    excluded.parquet        # 第一阶段排除ID与原因
    decisions.parquet       # 每条原始输入的一条二元决策
    manifest.json           # 指纹、数量和完成标记
  final/
    usable.parquet          # 清洗及标签均成功的最终数据
    excluded.parquet        # 含第一阶段和分层阶段排除原因
    decisions.parquet
    manifest.json
    dataset/                # Hugging Face Dataset，可直接传入现有训练入口
  logs/<时间-stage>/
    config.json
    pipeline.log
    gpu-0.log
    gpu-1.log
    swanlab.json
    summary.json
  summary.json
```

可用数据包含原始题目、答案、参考解答、稳定ID、重复组、数据版本、领域、易中难、标签来源。运行日志及汇总上传SwanLab，原始题目/逐题数据不作为SwanLab文本附件上传。

在本仓库使用当前首轮9,984题训练子集：

```bash
source preparation/env.sh
source preparation/pilot/profiles/balanced.env.sh
source preparation/data_engineering/splits.env.sh
bash preparation/run_math_opd.sh forward_kl_y_r --dry-run
```

原训练准备器已增加对稳定ID和标签的传递，后续y_o/y_r保留这些信息；独立脚本包不包含训练代码，若在旧仓库训练，应同步该适配改动。若最终可用数为0，脚本以失败退出且不创建可训练Dataset。

CPU回归检查：

```bash
python -m unittest discover -s preparation/data_engineering -p 'test_*.py' -v
```

数据划分脚本为`split.py`，已执行一次并冻结`data/processed/math-v1/splits-v1/`。验证集从终答一致的非空参考解答候选中按领域×难度选取；训练池固定顺序的前9,984题用于首轮，更长前缀用于后续扩充。当前训练配置及各集合路径由`splits.env.sh`导出。

已有只读数据画像及早期方案见`PLAN.zh-CN.md`和`audit-summary.json`；正式运行、实际划分及使用说明以`REPORT.zh-CN.md`为准；SwanLab链接继续记录在仓库`HISTORY.md`。
