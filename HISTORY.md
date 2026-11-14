# 探索与实验历史

## 2026-11-14：数学 OPD 复现准备

- 目的：复现 TRD 项目的数学 OPD，学生为 Qwen3-4B-Instruct-2507，教师为 Qwen3-8B；尚未运行正式完整训练。
- 分析：阅读 README、训练/评测脚本与依赖配置。原配置为 8 卡，本机为 2×96GB；梯度累积由 16 调整为 64，保持有效批量 128。FlashAttention 采用与现有 PyTorch 匹配的 ABI TRUE wheel。
- 数据：完整 DeepScaleR 40,315 条；AIME24/AIME25/HMMT25/BeyondAIME/AMO-Bench 分别为 30/30/30/100/39 条。AMO-Bench 按原脚本过滤描述题；学生和教师 token ID 映射一致。版本与校验记录见 `preparation/assets.json`。
- 修改：新增 `preparation/` 环境、下载、数据准备、启动和检查脚本；配置 checkpoint 与导出模型分盘保存及空间检查，未修改原训练算法。说明见 `preparation/README.zh-CN.md`。
- 检验：依赖、两卡 FlashAttention 前向/反向、NCCL、数学判分均通过；实际 Ray/vLLM 学生生成通过。两条合成样本完成一次两卡 TRD 全词表 KL 更新，两 rank 均通过，KL loss 为 1.4157005；此结果仅用于验证运行链路，不是正式复现指标。
- 证据：`preparation/readiness.json`、`preparation/validation.json`；日志位于 `preparation/logs/`，包括 `check-runtime.log`、`check-nccl.log`、`check-math-scoring.log`、`smoke-rollout.log`、`smoke-train.log`、`check-storage.log`。

## 2026-11-14：建立工作约定

- 按用户要求建立 `Agent.md`，明确工作目的、Agent 自主管理 Git，以及持续维护文字历史。
- 本次文档工作分支为 `docs/agent-workflow`。

## 2026-11-14：补充 SwanLab 记录约定

- 用户确认本地环境已登录 SwanLab；所有本地和远程实验均须上传 SwanLab，并在 `HISTORY.md` 中记录实验链接。
