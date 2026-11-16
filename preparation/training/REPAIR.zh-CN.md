# 缓存轨迹复用与聊天格式修复

本轮只做格式修复对照：从原始4B重新初始化LoRA，冻结8B；保留首轮9,984题的全部教师目标文本及think，不筛题。数学正确率效果由完整重训后的生成评测判断。

```bash
# 检查来源、划分、模型和输出计划，不运行GPU或创建实验
bash preparation/run_repair_train_eval.sh --run-id repair-chat-9984-v1 --dry-run

# 训练成功并合并LoRA后，自动执行AIME25、pass@16、TP2
bash preparation/run_repair_train_eval.sh --run-id repair-chat-9984-v1

# 仅训练
bash preparation/run_repair_train.sh --run-id repair-chat-train-only
```

已有run-id不能覆盖，重复运行需使用新ID；入口有双卡互斥锁。默认生成来源固定为`preparation/logs/math-runs/20261115-185850-279780-53607/train/summary.json`；可用`REPAIR_CACHE_SOURCE`指定另一份完成记录，但模型和冻结划分必须相同。此参数引用数据来源，不恢复旧权重或优化器。缓存不完整或不一致直接报错，不启动生成服务。

## 修复内容

- 师生分别应用各自的chat template，复用各自在生成时的原始user提示，保留教师额外看到学生初稿的设计。双方只在同一目标response的因果预测位置计算KL，不把初稿与改写逐字对齐。
- 只渲染user与assistant起始提示，直接编码缓存response；不让教师模板重写think内部空白。词表token ID和结束token必须一致。
- 旧缓存没有finish_reason，结束状态采用保守推断：接近16,384 token上限、think未闭合或末boxed无法解析的52条不额外添加EOS，但仍参与response监督；其余9,932条添加结束token。完成不代表答案正确。
- 原始缓存只读；每次在`train/cache/chat-training.parquet`派生输入，保留9,984行、顺序和response文本，追加学生原始提示与结束状态。manifest记录原summary、两份缓存及派生输入的SHA256。
- 训练前检查完成状态、原始4B/8B路径、冻结划分manifest、训练/验证身份隔离、教师缓存与源记录哈希、逐题初稿与initial_response一致、师生ID/题目/gold一致及原教师提示与该题关联。
- 修复入口启用`SEQUENCE_FORMAT=chat`；旧入口默认plain，用于保留复现口径。旧512题参考PPL/KLD协议不被静默修改。

## 容量与记录

本次同时保留上一轮模型。通用门槛仍为checkpoint盘20GiB、导出盘18GiB；修复入口的`balanced-single-save`仅对已实测的4B/BF16/LoRA64、有效batch128、1 epoch/78更新且不周期保存中间全量checkpoint的配置，使用12GiB/11GiB门槛，同盘时相加。参考首轮实测8.971GiB和8.478GiB；超出配置范围会拒绝使用该门槛。

SwanLab继续使用trd-math-opd，训练与评测分别记录，失败保留。配置保存代码和数据哈希；模型、缓存、逐题产物及原始日志不提交Git。

## 已完成的检查

- [CPU检查](https://swanlab.cn/@WangZIyu/trd-math-opd/runs/rcywdvtl)：46项回归通过；完整派生9,984题，随机512题师生因果目标一致，0条目标尾部裁剪。
- [双卡短程检查](https://swanlab.cn/@WangZIyu/trd-math-opd/runs/y86py49p)：balanced配置、64微批/rank、2个优化器更新、256次样本访问，两rank完成且记录数值有限。NVML峰值81.54/57.44GiB；不保存该试跑权重。
- [最终回归与dry-run](https://swanlab.cn/@WangZIyu/trd-math-opd/runs/bf3w6lhm)：47项回归通过，包含容量配置约束；训练及两套评测入口dry-run通过。
- 开发前两次CPU检查在测试夹具阶段失败，原因分别为Mock缺少可序列化url、夹具缺reward_model；均未启动GPU，保留[首次](https://swanlab.cn/@WangZIyu/trd-math-opd/runs/mfo48l7h)和[第二次](https://swanlab.cn/@WangZIyu/trd-math-opd/runs/l3darndn)失败记录。

复核入口：先加载`preparation/training/config.env.sh`并设`SEQUENCE_FORMAT=chat`，运行`python preparation/training/run_repair_checks.py`；通过后用`python preparation/training/run_repair_smoke.py --validated-run <CPU检查summary.json>`执行双卡短程验证。所有实际检查自动在线记录。

本地检查目录为`preparation/logs/training-development/chat-repair-cpu-20261116-070052`及`chat-repair-gpu-20261116-070304`，最终回归日志位于同根目录；小型摘要见[repair-validation.json](repair-validation.json)。

后续去除教师改写思考监督与可靠终答筛选分别作为独立对照，保留当前格式修复结果，以便判断收益来自哪项修改。

正式格式对照已于2026-11-16 07:13 UTC启动，run-id为`repair-chat-9984-v1`，[训练SwanLab](https://swanlab.cn/@WangZIyu/trd-math-opd/runs/qejfvx16)。控制器日志为`preparation/logs/repair-launchers/repair-chat-9984-v1.log`，训练状态为`preparation/logs/math-runs/repair-chat-9984-v1/train/summary.json`。训练与合并成功后自动启动AIME25评测；实际完成状态及成绩以各阶段summary为准。

用户随后将本轮范围缩减为仅AIME25。运行根目录的evaluation-control.json在生成前跳过AIME24；已加载旧两套汇总逻辑的控制器可能在收尾时报缺少AIME24，后台finish_reduced_eval.py等待其自然退出后，仅核对AIME25全部30×16响应、恢复同一SwanLab记录并生成最终summary。旧summary及日志保留，训练进程不接收信号。新启动入口直接按AIME25汇总，不需要该兼容收尾。控制器状态见evaluation-scope-status.json。
