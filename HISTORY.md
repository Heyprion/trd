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

- Pilot `baseline-2k`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/8dnul6rc；配置/日志/显存采样：`preparation/logs/pilot/baseline-2k`（结果见该目录 summary.json）。

- KL 数值/梯度一致性检验：https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/gc4jqjqm；`preparation/logs/pilot/numerical`。

- Pilot `baseline-18k`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/a906hp9e；配置/日志/显存采样：`preparation/logs/pilot/baseline-18k`（结果见该目录 summary.json）。

- KL 数值/梯度一致性检验：https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/ac2k0sp6；`preparation/logs/pilot/numerical`。

- Pilot `recompute-18k`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/50oqfkd1；配置/日志/显存采样：`preparation/logs/pilot/recompute-18k`（结果见该目录 summary.json）。

- OPD training：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/7meyjp7e；日志：`/root/autodl-tmp/trd/preparation/logs/pilot/recompute-18k`。

- Pilot `resident-18k`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/drsrf2ju；配置/日志/显存采样：`preparation/logs/pilot/resident-18k`（结果见该目录 summary.json）。

- 生产 autocast 环境全词表数值/梯度检验：https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/11g4igj4；`preparation/logs/pilot/autocast-numerical`。

- Pilot `resident-18k-b2`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/vzqyy8e1；配置/日志/显存采样：`preparation/logs/pilot/resident-18k-b2`（结果见该目录 summary.json）。

- 生产 autocast 环境全词表数值/梯度检验：https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/wa6axwl3；`preparation/logs/pilot/autocast-numerical`。

- Pilot `resident-18k-b3`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/4bol3f34；配置/日志/显存采样：`preparation/logs/pilot/resident-18k-b3`（结果见该目录 summary.json）。

- Pilot `bf16-18k-b2`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/7dlotxzz；配置/日志/显存采样：`preparation/logs/pilot/bf16-18k-b2`（结果见该目录 summary.json）。

## 2026-11-14：双 RTX PRO 6000 数学 OPD 首轮性能试跑（进行中）

- 用户要求：保持单机双卡 FSDP2，少量长短样本测试超参、显存与速度；目标每卡整卡占用约不超过 90 GiB，保留合理余量。当前分支 `perf/math-opd-pro6000-pilot`。
- 已核实：GPU 各 97,887 MiB，拓扑 NODE/PCIe、无 NVLink；原始入口使用 LoRA r64/alpha128、学习率 5e-6、全词表 KL。性能试跑不评价收敛，不用少量样本调整学习率、LoRA rank 或温度。
- 测试方式：`preparation/pilot/prepare_stress.py` 从 DeepScaleR 前 32 条的真实解答构造固定长度压力输入（必要时重复文本），真实 token 数记录在每 rank 指标中；不把这些输入当作真实 on-policy 样本。`run_probe.py` 父进程以 0.1s 间隔采样 NVML 两卡整体显存、利用率和功率，子进程实际运行 FSDP2 师生前向、KL 反向、优化器更新。所有试验均在线上传 SwanLab，失败也保留记录；链接见上文各条。
- 初始结果：原实现 2K/18,432 tokens、batch=1 均通过；18K 预热后整步约 6.68s，NVML 峰值 59.79 GiB。逐样本重计算 KL v1 为约 6.04s / 58.62 GiB；将 FP32 教师常驻 GPU 后约 5.99s / 73.88 GiB，收益很小。FP32 常驻教师、batch=2/3 均 OOM（已通过 torchrun 终止同组进程并上传失败日志）。因此不采用单纯关闭教师 offload 的配置。
- 查明两个实现细节：verl 即使外部 offload=false 也会对 forward_only 引擎强制 CPUOffloadPolicy；FSDPEngineConfig.model_dtype 默认 fp32，使冻结教师也以 FP32 保存。use_torch_compile 实际只编译 entropy，而本实验关闭 entropy，不能把此开关作为加速收益。
- KL 数值检查：普通上下文 24 组前向/梯度测试通过；进一步在 151,936 全词表及 BF16 autocast 下用稳定 FP64 参考检查。原 forward KL 的 einsum 受 autocast 降精度影响，近似相同的两分布测试中梯度相对误差约 8.19%；重计算版约 0.0304%，前向最大绝对误差 <1.5e-6。故重计算版保持数学全词表 KL 目标，但不承诺与原 BF16 einsum 实现逐位一致。最早一次 FP32 einsum 参考测试失败是该参考本身长归约误差，需要使用稳定 FP64 参考；失败已记录 SwanLab。
- 后续实现：新增打包后的整批 KL 重计算 v2，避免逐行 nested unbind 反向对全 logits 反复分配梯度；测试 BF16 冻结权重存储（LoRA 优化器参数精度单独核实）、reshard、Liger 及梯度检查点选项。自 v2 起每次 probe 保存源码快照和工作区 diff，支持复核。
- 参考：Liger Kernel 论文 https://arxiv.org/abs/2410.10989 （分块和融合减少中间量）；PyTorch 2.8 FSDP2 文档 https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html （reshard 的内存/通信权衡）。未将论文中的加速百分比当作本机实测。

- Pilot `bf16-18k-b2-noreshard`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/74b8x163；配置/日志/显存采样：`preparation/logs/pilot/bf16-18k-b2-noreshard`（结果见该目录 summary.json）。

- Pilot `bf16-18k-b2-liger`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/5ptdslru；配置/日志/显存采样：`preparation/logs/pilot/bf16-18k-b2-liger`（结果见该目录 summary.json）。

- Pilot `bf16-18k-no-gc`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/5q7qz65b；配置/日志/显存采样：`preparation/logs/pilot/bf16-18k-no-gc`（结果见该目录 summary.json）。

- Pilot `bf16-2k-b8`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/wiuz66io；配置/日志/显存采样：`preparation/logs/pilot/bf16-2k-b8`（结果见该目录 summary.json）。

- Pilot `offload-liger-18k-b2`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/alb15fup；配置/日志/显存采样：`preparation/logs/pilot/offload-liger-18k-b2`（结果见该目录 summary.json）。

- Pilot `resident-liger-18k-b1`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/ieert45i；配置/日志/显存采样：`preparation/logs/pilot/resident-liger-18k-b1`（结果见该目录 summary.json）。

- Pilot `resident-liger-16k-b2`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/1gjsnvpg；配置/日志/显存采样：`preparation/logs/pilot/resident-liger-16k-b2`（结果见该目录 summary.json）。

- Pilot `resident-liger-2k-b16`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/mumtgki0；配置/日志/显存采样：`preparation/logs/pilot/resident-liger-2k-b16`（结果见该目录 summary.json）。

- Pilot `capped-offload-18k-b2-acc4`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/hdw9dx5x；配置/日志/显存采样：`preparation/logs/pilot/capped-offload-18k-b2-acc4`（结果见该目录 summary.json）。

- Pilot `dynamic32k-resident-18k-b2`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/vbcosoyz；配置/日志/显存采样：`preparation/logs/pilot/dynamic32k-resident-18k-b2`（结果见该目录 summary.json）。

- Pilot `sp2-resident-18k-b1`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/wia4p2lb；配置/日志/显存采样：`preparation/logs/pilot/sp2-resident-18k-b1`（结果见该目录 summary.json）。

- 生产 autocast 环境全词表数值/梯度检验：https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/3ro3hcoy；`preparation/logs/pilot/autocast-numerical-cuda`。

- Pilot `capped-resident-16k-b2`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/7dqnisz5；配置/日志/显存采样：`preparation/logs/pilot/capped-resident-16k-b2`（结果见该目录 summary.json）。

- Pilot `capped-resident-2k-b16`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/tqvk3qz1；配置/日志/显存采样：`preparation/logs/pilot/capped-resident-2k-b16`（结果见该目录 summary.json）。

- 真实 OPD/TRD rollout：https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/2enpetf3；`preparation/logs/pilot/real-rollout`。

- Pilot `real-opd-trd-update`：SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/ddcwfrbe；配置/日志/显存采样：`preparation/logs/pilot/real-opd-trd-update`（结果见该目录 summary.json）。

- OPD training：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/fxue8aiu；日志：`/root/autodl-tmp/trd/preparation/logs/pilot/real-opd-trd-update`。

## 2026-11-15：双卡数学 OPD 性能试跑完成

- 共 21 个训练 probe：17 个通过，4 个探索配置 OOM；失败已上传 SwanLab。GPU 实际空闲收尾，未启动完整数据训练。配置、完整结果与限制见 `preparation/pilot/REPORT.zh-CN.md`、`results.json`、`validation.json`；逐步执行和所有链接保留在上文。
- 推荐 balanced：FSDP2 size=2、SP=1、per-rank batch=2、累积=32（有效 batch=128），max_length=18,432、动态微批预算=32,768，BF16 冻结权重/FP32 LoRA 参数，Liger、梯度检查点、KL 重计算 chunk512、reshard=true、教师 GPU 常驻、CUDA allocator fraction=0.92。源脚本 `preparation/pilot/profiles/balanced.env.sh`，原入口 dry-run 已验证参数传递。两条完整 18K 自动拆微批：峰值 76.72 GiB / 吞吐 7.08K tokens/s，约比原 18K 基线高 29%；填满 32K 预算的实际测试：峰值 87.93 GiB。
- 高容量备选 `profiles/high_memory.env.sh`：预算 36,864、教师 CPU offload，两个完整 18K 连续 8 微批/两次累积更新，通过，峰值 88.44 GiB / 吞吐 6.28K；比 balanced 慢。2K 短序列 batch16、32K预算且 allocator=0.92：峰值 88.06 GiB / 吞吐 8.97K（原 batch1 约 2.87K），仅适用短样本子集。
- SP=2 虽然壁钟单次更短，但去除重复 SP 数据后总吞吐只有 4.74K，比 SP=1 的约 7.1K 低，故采用 SP=1。更大 batch、不重分片、关闭梯度检查点的失败配置不采用。
- 新 KL 路径 CUDA 全词表/BF16 autocast 对照稳定 FP64 通过：前向最大误差 <1.2e-6；近相同分布 forward KL 梯度相对误差新路径约0.012%、旧 einsum 约8.03%。保留旧后端默认，只在显式 profile 选择新路径；不声称逐位复现旧 BF16 数值。
- 8 题真实 OPD/TRD 生成/改写通过（输出 cap1024，冷启动共 296.9s，峰值 79.74 GiB）：https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/2enpetf3。随后真实数据两卡更新及原生 train_epoch 通过，每 rank 4 次优化器更新（2 probe + 2 native）；SwanLab https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/ddcwfrbe 和 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/fxue8aiu。原始 parquet、配置、逐 rank 数值和日志见 `preparation/logs/pilot/real-rollout/`、`real-opd-trd-update/`。
- 核心实现提交 `d977a7d`，原生循环释放旧 teacher logits 与完整启动参数记录提交 `b833978`。所有原始压力日志在 `preparation/logs/pilot/`，后期有 source 快照、数据 SHA256 和工作区 diff；初始逐行 KL v1 与最终 packed v2 的区别已在报告说明。
- 本轮未做收敛、正式准确率、完整16K输出长度 rollout 或完整128有效批量长程验证；长序列累积验证为2/4，后续完整运行需继续观察稳定性和质量。

- 首轮完整报告、候选对比与两套推荐配置已上传 SwanLab 汇总：https://swanlab.cn/@WangZIyu/trd-pro6000-pilot/runs/u6ewxwcl。

## 2026-11-15：基于已完成试跑估算全量耗时（未重新运行）

- 用户要求：仅按已有试跑预测完整数学训练和最终推理测试耗时，不再启动试验。只读取已有配置及日志，没有启动训练、生成或评测。
- 范围：单个 forward_kl_y_r/TRD 实验，DeepScaleR 40,315 条（长度过滤可能稍减），1 epoch，全局 batch128，约315次优化器更新；最终评测五套共229题，每题16次，共3,664份输出，只评训练后的学生模型。
- 依据：balanced 约10,079个外层微批；若全部近18K，每微批约10.33s，纯KL约29h；考虑实际较短输入，纯KL粗估10–30h。已有8题/1K生成日志中学生实际生成约9.6s、教师约14.9s，另有近5分钟总冷启动；不能直接按含冷启动的296.9s等比例放大。全量并发与长输出未实测，生成侧预测置信度较低。
- 用于排期的粗估：学生全量生成5–15h，教师全量改写40–90h，KL更新10–30h，训练全流程合计约55–135h；最终五套pass@16推理与判分约3–10h；合计约3–6天，中心预算约4天。假设平均生成长度数千至约一万token，未把16K训练生成上限和38,912评测生成上限当作平均长度；大量输出触顶会延长耗时。
- 更正上轮长度表述：原TRD wrapper 的 MAX_LENGTH 默认34,816，本轮profile是18,432，属于截断策略变化而非仅调整硬件参数。上述耗时对应当前修改后的profile；原34,816严格长度复现尚无显存/速度验证，不应声称已完成对应调优。报告已补充此边界，未修改训练配置。

## 2026-11-15：将双卡优化配置确认为暂定默认

- 用户确认采用已调好的双卡 balanced 配置，并要求简短、全面地写入 `Agent.md`，供后续会话默认使用。
- 新增 `Agent.md` 第5节：记录模型/方法、FSDP2与SP、batch/累积、序列上限与动态token预算、精度及GPU优化、offload/显存策略、LoRA与学习率、完整数据和生成/评测设置，以及加载环境和profile的标准启动命令。
- 明确以 `preparation/pilot/profiles/balanced.env.sh` 为配置来源；后续启动默认显式加载它，覆盖 `env.sh` 的旧 batch=1/累积=64。保留18,432相对原34,816的长度差异说明，试跑子集和1,024输出上限不作为正式默认。
- 本次只更新文档，核对配置文件并执行文本差异检查；未启动训练、生成、评测或新的实验。

## 2026-11-15：保存并同步当前工作到远程仓库

- 用户要求将当前 Git 改动全部保存并上传远程。同步前工作区已干净，既有代码和文档均已提交；已成功 fetch 并核对 origin 为 `https://github.com/Heyprion/trd.git`。
- 本次同步分支为 `perf/math-opd-pro6000-pilot` 与 `docs/agent-workflow`，覆盖环境准备、Agent约定、双卡优化、试跑工具/结果、默认配置和历史记录的全部新增提交；使用普通非强制推送并建立上游跟踪。
- 本次属于 Git 同步，无新增训练、生成或评测实验；模型/数据和原始日志按既有忽略规则保留本地，实验日志及结果链接继续由 SwanLab 和 HISTORY 索引。

- 同步方式补充：普通 HTTPS push 因缺少终端凭据失败，未创建任何远程分支；已确认连接的 GitHub 账号 Heyprion 具有本仓库写权限，改用该连接的 Git 对象接口逐次上传。接口不支持原 author/committer 时间，远程 SHA 将不同；逐提交校验文件树一致，保留原始 Git bundle、提交对象和 SHA 映射，本地保留 archive 分支。此补充取代上文“普通推送”的执行方式描述。

- 用户补充上传边界：只上传代码、文档、配置和小型结果摘要；模型/数据/缓存等大文件不得上传。核对确认这些目录均被 Git 忽略，尚未上传任何文件。历史备份已移入本地 `.git/codex-sync-backups/`，不会上传备份包或备份元数据；待上传内容逐项检查文本类型和大小。远程重建提交时跳过仅涉及备份包的本地提交，所有代码和文档版本保留。Agent.md 已记录此约定。

- 远程同步核验完成：`docs/agent-workflow` 和 `perf/math-opd-pro6000-pilot` 已通过 GitHub 连接创建；10 个对应提交逐一核对文件树完全相同，本地工作分支与远程已对齐。上传内容为53个代码/文本文件、约310KB，最大约51KB；未上传模型、数据集、环境、缓存、checkpoint、原始日志或 Git 备份包。原始提交备份、SHA映射及归档引用仅保存在本地 `.git/codex-sync-backups/`、`refs/codex-local-archive/`。

## 2026-11-15：制定蒸馏前后PPL/KLD对比方案（未启动评测）

- 用户要求：解释PPL/KLD应在哪类数据上测，给出完整比较方案与耗时预估；暂不做最终数学测试集推理评测。本次只读取源码、配置、旧日志和数据，并进行CPU token长度画像，没有启动训练、生成或模型评分实验，因此没有新增SwanLab实验链接。
- 当前状态：核对只有性能/数值/少量真实数据试跑，`checkpoints_merged/`及`/root/trd-checkpoints/`无模型文件，尚无正式全量蒸馏后的模型可直接比较。原训练器已有训练批次的学生/教师PPL及KL，但它们不能替代固定验证集的checkpoint前后对比。
- 数据检查：本地DeepScaleR 40,315条，非空参考`solution` 7,391条、空/缺失32,924条。最初对无筛选512题检查发现大量空解答，随后改为非空子集画像。从索引>=32的非空解答用NumPy default_rng(42)无放回抽512题，题目平均83.8 tokens；解答平均413.4、中位373、P95 844.3、最大2,499、合计211,681 tokens。仅做描述性抽样，尚未冻结验证集或修改训练输入。
- 方案：主结果建议用真正未参与蒸馏的512题固定参考解答，对基线4B、正式蒸馏后4B及固定8B教师做teacher-forcing；记录response NLL/PPL与相同输入下全词表forward KL(T||S)，可同时补reverse KL、长度分桶及按题配对置信区间；另用128题训练对照辅助判断拟合局限。明确有参考解答子集的覆盖边界。留出512题会改变完整40,315条复现默认，本次未执行；严格全量复现需另建去重外部验证集。
- TRD特殊口径：当前学生看题目、教师额外看固定初始解答及修订指令，训练KL是跨条件目标。独立64–128题的固定y_o/y_r分析可选补充，单独命名，不与相同输入主表混合；不同模型各自输出上的PPL不作为前后主对比。
- 排期：既有约7.1K训练response tokens/s不能直接作为三模型评分实测。按暂定1–4K固定参考位置/s合计评分吞吐，512题主表前后加教师预计10–30分钟，加128题训练对照和汇总15–40分钟；可选64–128题TRD长轨迹含生成0.5–2小时。首次脚本开发与核验另计2–4小时，先32题校准更新预算。所有未来评分/生成实验均须上传SwanLab并回写链接。
- 文档与参考：`preparation/PPL_KLD_PLAN.zh-CN.md`；Hugging Face https://huggingface.co/docs/transformers/perplexity ，TRD https://arxiv.org/html/2606.08432v1 ，本地`recipe/opd/kl_trainer.py`和`dataset/data_utils.py`、pilot结果与日志。工作分支`docs/math-ppl-kld-plan`，本轮只提交方案和历史。
