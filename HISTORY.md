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

## 2026-11-15：核实当前验证集与测试集规模

- 用户追问当前项目是否已有验证集及其规模。读取实际数据文件元数据、DeepScaleR准备脚本及OPD训练/评测配置；当前没有单独划分并接入训练流程的PPL/KLD验证集，已落地规模为0，前文512题仍是待执行方案。
- 当前DeepScaleR训练文件为40,315条，准备脚本默认只读取train split。`data/eval/`现有的是最终数学benchmark测试数据：AIME24 30、AIME25 30、HMMT25 30、BeyondAIME 100、AMO-Bench 39，合计229题；AMO-Bench原始50题与实际过滤后39题应区分。没有启动任何评测或修改数据划分，无新增SwanLab实验。

## 2026-11-15：核查TRD论文公开权重

- 用户询问论文是否提供训练前base、教师及蒸馏后学生权重。在线核查最新论文v2 https://arxiv.org/html/2606.08432v2 、作者仓库 https://github.com/louieworth/trd 的README/完整文件树/Releases/Issues，以及HF论文页 https://huggingface.co/papers/2606.08432 。作者仓库当前无Release，文件树无模型权重，README无蒸馏后模型下载链接；HF论文页关联模型为0，公开检索未找到作者发布的TRD学生checkpoint或LoRA adapter。结论限定为截至本日上述公开渠道未发现，不能据此断言作者没有私有权重。
- 补充证据：作者仓库Issue #1 https://github.com/louieworth/trd/issues/1 标题为“Release TRD-distilled models and AMOBench on Hugging Face”，2026-06-09提出发布请求，目前仍open、0评论。该issue提及的旧README“Coming soon”不在当前README中，不将其误写为当前公告。
- 可用初始化权重为Qwen官方 https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 和 https://huggingface.co/Qwen/Qwen3-8B 。本地已分别有3个/5个safetensors分片，约8.04GB/16.38GB；对应路径`models/Qwen3-4B-Instruct-2507`、`models/Qwen3-8B`，revision与校验见`preparation/assets.json`。论文Base是相对本次蒸馏的起点，当前4B起点本身是Instruct模型。
- 方法澄清：当前OPD使用冻结8B教师，只更新4B学生，因此不存在需要另取的“本次蒸馏后教师权重”；蒸馏后学生仍需自行训练或取得作者checkpoint，不能从结果表恢复。只做在线资料核查和本地权重文件清点，未下载新权重、未启动实验，无新增SwanLab运行。


## 2026-11-15：数据工程只读审计与清洗/难度分层方案

- 用户要求：基于真实数据抽样和相似论文/工程，提出清洗筛选、难度分层方案，重点限制本机双卡成本。本次只读审计和写方案，未覆盖原始/训练数据，未运行 GPU 标注、生成或训练。
- 数据核查：读取 DeepScaleR 完整40,315行（revision b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29），确认本地训练parquet逐行等于raw JSON；固定seed=20261115浏览64题并查看风险候选。空answer=6、空solution=32,924，后者为数据集原约定，当前OPD不使用参考解答，不能据此剔除。题干中位65 tokens、P95约179，实际学生chat prompt仅1条超过2048。
- 去重：原文完全相同冗余928条；NFC/去空白后950组、冗余1,282条；177组答案文本不同，包含等价格式和真实错标。已核实原始行786/1401同一道合作做纸杯蛋糕题答案分别25/20，正确为25且两条solution均算到25。样本还发现缺图和解答片段混入problem，关键词仅作风险标记。
- 测试排重：读取五套实际评测共229题，规范化全等0对；trigram Jaccard>=0.5召回1对，已阅读核实训练行5696与BeyondAIME索引30为同一二十面体染色题，测试仅追加答案mod1000（61917364224 ->224）。方案要求从clean训练版本移除整个题组；本轮尚未删除，不把零全等称为无污染。
- 标签恢复：只读下载官方Omni-MATH，固定revision 40ba231d8f16e29ecd40e6407e2c8640145a8f62；4,428行与本地3,528训练行匹配，共3,587匹配对。3,469训练行单一匹配，3,468同时通过现有答案判等；59行多匹配、31行难度冲突，需保留不确定状态。来源文件仅保存在data/raw/KbsdJames/Omni-MATH/，未加入Git。
- 审计与记录：v1 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/1hc7vi14；v2补充一对多及冲突统计 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/ksp7x0ba。v2分析段约19.4秒（不含导入/网络连接），核心统计与v1一致。原始日志、64题样本、候选、逐题tokens和join保存在preparation/logs/data-engineering/audit-v2/，stdout为preparation/logs/data-engineering-audit-v2.log。现有grader未判等不能当成错标，审计未证明完整语义去污染。
- 方案：CPU规则清洗/哈希及近重复召回/测试排重 -> 原标签回填 -> 8B非思考短结构标注补齐先验难度/领域/风险 -> 复用正常y_o/y_r做显式数学判分与行为分组。生成入口当前不主动评分，默认reward=0不可当作正确率；需补保留finish_reason等元信息。先验难度、模型表现和长度成本独立保存，不照搬DAPO的全对/全错删除规则。首轮保持清洗后自然分布，不预设课程比例；允许unknown并分流quarantine。
- 成本：未来短标注拟双卡各1个8B vLLM副本、TP1，沿用balanced rollout显存份额0.85/并发16为起点；先256题测速后再全量。首版GPU预留1–3小时，可选256题×2次4B解题校准另加0.5–1.5小时；均为未实测工程估计，开发联调另计半天到一天。全量4次长解题仅按平均2K输出就约3.23亿tokens，故不作首版。与已有PPL/KLD方案对齐的512参考验证题仍为待执行建议，当前没有新划分。
- 产物与参考：preparation/data_engineering/PLAN.zh-CN.md、audit.py及小型audit-summary.json；参考DeepScaleR原卡、Big-Math、Omni-MATH、Open R1 Update #3、DAPO与Qwen3-8B官方卡，具体适用边界和链接在方案中。工作分支analysis/data-cleaning-stratification；提交前检查只含代码/文档/小摘要，不含数据与原始日志。
- 检验：审计脚本导入/语法检查通过，汇总行数、答案类型、非空解答、评测规模及来源匹配计数恒等式通过，规范化保留数值及不等号的检查通过；git diff --check通过。数据、原始日志与__pycache__均由Git忽略。后续生成目录按数据版本隔离并校验输入哈希的要求已加入方案，避免旧parquet缓存误复用。


## 2026-11-15：按用户要求收紧数据工程为一小时全自动流程

- 用户明确纠正：不安排人工核查；每条数据最终只有可用/排除；难度只有易/中/难；缺图、缺条件、缺答案和答案泄露在清洗阶段处理，不能混入分层；原始标签冲突直接不用。清洗整体希望控制在1小时以内，充分利用双卡、大内存和并行worker。
- 已重写preparation/data_engineering/PLAN.zh-CN.md，替代上一版三池/四档及人工复核方案。基础规则和精确重复由CPU处理，近重复由CPU召回、GPU批量判同题；双卡各一个8B BF16/TP1副本，先短输出内容清洗，再仅给可用题回填可靠原标签或打领域/易中难标签。空solution符合数据约定仍保留，空answer排除；答案或来源标签冲突无法自动闭环时排除整个题组。
- 并行安排：CPU初始16 workers，重操作按吞吐增至32；每卡max_num_seqs由64测速到128、token批预算32K视吞吐测速64K，显存份额0.85、非思考、前缀缓存、常驻模型。当前进程可见208逻辑CPU、约1TiB系统内存；不能把增加worker当作无限加速。精确哈希保留CPU依据是先前整个分析段仅19.4秒，尚无GPU迁移更快证据；GPU用于模型判断和去重候选判定。
- 预算：清洗和标签各最多8输出tokens，两轮上界约64.5万tokens；两卡合计500–2000token/s假设下纯decode约5.4–21.5分钟。端到端60分钟目标包含加载、256题测速、CPU/GPU处理、导出与上传，需首轮实测确认；到45分钟不追加复杂判断，55分钟未闭环记录按技术超时排除并明确计数。移除完整学生解题校准和题目修复，不在本轮新增验证集划分。脚本首次开发时间单计。
- 本轮只修订方案、查询CPU/内存和本机vLLM0.11.0官方并行/调优文档，没有启动GPU或其他实验，无新增SwanLab链接。上文审计记录继续有效；实施阶段按新方案执行，原始数据未改动。

## 2026-11-15 08:29:50 UTC：数据工程 all

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/hrovl3ci；配置及日志：`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-v2/logs/20261115-082947-all`；输出：`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-v2`。
- 完成：输入26，可用20，排除6，技术排除0，耗时24.5s；汇总`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-v2/logs/20261115-082947-all/summary.json`。

## 2026-11-15 08:33:36 UTC：数据工程 all

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/z69empw0；配置及日志：`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-v3/logs/20261115-083332-all`；输出：`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-v3`。
- 完成：输入30，可用21，排除9，技术排除2，耗时24.2s；汇总`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-v3/logs/20261115-083332-all/summary.json`。

- 数据工程全量CPU规模验证：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/v96h280a；日志/汇总：preparation/logs/data-engineering/script-smoke/cpu-scale。

## 2026-11-15 08:37:00 UTC：数据工程 all

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/qjk4letv；配置及日志：`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-v4/logs/20261115-083656-all`；输出：`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-v4`。
- 完成：输入30，可用23，排除7，技术排除0，耗时25.5s；汇总`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-v4/logs/20261115-083656-all/summary.json`。

## 2026-11-15 08:38:46 UTC：数据工程 label

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/fpcewyw1；配置及日志：`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/split-stage-validation/logs/20261115-083842-label`；输出：`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/split-stage-validation`。

## 2026-11-15 08:38:46 UTC：数据工程 all

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/hnmrdyc0；配置及日志：`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-v4/logs/20261115-083842-all`；输出：`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-v4`。
- 已核对并复用完成产物：/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-v4/final/manifest.json；未运行GPU。
- 完成：输入30，可用23，排除7，技术排除0，耗时23.0s；汇总`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/split-stage-validation/logs/20261115-083842-label/summary.json`。

## 2026-11-15 08:40:27 UTC：数据工程 all

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/yhe3bei5；配置及日志：`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-final/logs/20261115-084024-all`；输出：`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-final`。
- 完成：输入30，可用23，排除7，技术排除0，耗时23.8s；汇总`/root/autodl-tmp/trd/preparation/logs/data-engineering/script-smoke/run-final/logs/20261115-084024-all/summary.json`。


## 2026-11-15：实现可搬运的一键清洗与分层脚本

- 用户要求：两个阶段分别提供脚本，并提供一键启动，方便在其他位置执行。已实现clean.py、stratify.py及统一run.sh/pipeline.py；路径和双卡参数集中在config.env.sh。默认完整DeepScaleR，all模式总预算3600秒/清洗1800秒/导出缓冲180秒，独立两个阶段默认各1800秒；模型和原数据均不随代码包上传。
- CPU逻辑：字段/编码/明显解答片段过滤、保守答案规范化、原来源匹配冲突整组排除、精确去重、MinHash/LSH及测试题倒排召回。保留数学符号、Unicode变量及绘图代码换行，空solution不删；每条输入必须有唯一usable/excluded结果，技术失败独立统计。
- GPU逻辑：两个独立BF16 Qwen3-8B副本、TP1，公共前缀缓存、默认eager以减少冷启动；清洗、近重复判断、分层分别使用有限枚举约束，最多8输出tokens，格式/推理错误最多短重试一次。数据量足够时比较有效batch64/128；小样本分发也实际使用两卡。分层用语义明确的短英文枚举，保持易/中/难三档，可靠原标签优先回填。第一轮短数码标签出现领域过度归入代数、解答片段漏检，已通过明确题型示例、保留语义的枚举和CPU规则修正；这不构成全量标签准确率保证。
- 产物：clean和final的usable/excluded/decisions parquet、完成manifest、HF Dataset训练目录、配置及逐GPU日志、每请求输出码和finish_reason。父进程总时限、输出目录锁和完成阶段输入/测试/来源/代码/模型配置及权重文件元数据指纹校验已接入；--resume只复用完成阶段。修改y_o_prepare.py传递稳定ID、版本及领域/难度，y_r继续继承extra_info。实际实验自动在线SwanLab记录和追加HISTORY。
- 启动兼容性：首次启动发现本机SwanLab0.10.5把SWANLAB_PROJECT平面环境变量当作嵌套JSON解析；该次在模型/实验初始化前失败，日志preparation/logs/data-engineering/script-smoke/launcher-v1.log。脚本已捕获项目名并在SDK初始化前移除此冲突变量。开发试跑v2（26题）https://swanlab.cn/@WangZIyu/trd-math-opd/runs/hrovl3ci；v3（30题、两条完整标签被尾随空白触发length而误当格式失败）https://swanlab.cn/@WangZIyu/trd-math-opd/runs/z69empw0；现按完整有限枚举解析判断有效，逐请求记录finish_reason。
- GPU验证：v4 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/qjk4letv；最终代码主要路径重跑 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/yhe3bei5，30输入→23可用/7排除，技术排除0，4条native+19条模型标签；easy/medium/hard为6/10/7。两GPU实际分别处理23/21个请求，含近重复/清洗/分层；模型加载约12.5–13.0秒，处理至导出约23.79秒，最终同步和退出另计。样本含已知答案冲突、缺图、解答片段、测试同题、重复和空答案，均按规则排除。原始样本和日志仅在preparation/logs/data-engineering/script-smoke/。
- 全量CPU规模验证：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/v96h280a，40,315输入→38,839个代表/1,476初始排除；召回589对候选（含2对评测候选），无候选预算/超大桶排除，约11.23秒。这是CPU检查，不是已完成全量GPU清洗，不对候选做未实测的同题断言。
- 独立分层入口验证：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/fpcewyw1，读取复制的clean阶段产物、显式传入不存在的原始input路径仍可完成，说明阶段2不依赖原始输入；23条可用、0新增排除。完成阶段--resume验证 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/hnmrdyc0 通过，无新增GPU工作。
- 检验：12项CPU回归检查、三入口dry-run、shell语法、训练metadata传递、Git diff检查通过。GPU均已退出并释放；未启动全量GPU数据工程、正式生成或训练。配置、README及小型VALIDATION.json入Git，原数据/逐题产物/日志不入Git。脚本包仅含代码、配置、文档和小摘要。
- 可搬运代码包已生成：`/mnt/data/math-data-engineering-scripts.zip`，38,078字节，SHA256 f80b22666b2933a5bb71f79f2b1dc37c98fa0afa739ac27dfc4b3dd541390d36。解压到独立/tmp目录后，一键入口dry-run与12项CPU回归检查均通过。包内仅15个代码/配置/说明/小摘要文件（含可选训练metadata适配patch），没有数据、权重、缓存或原始日志。

## 2026-11-15 08:56:26 UTC：数据工程 all

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/oqwvc29e；配置及日志：`/root/autodl-tmp/trd/data/processed/math-v1/logs/20261115-085622-all`；输出：`/root/autodl-tmp/trd/data/processed/math-v1`。
- 完成：输入40315，可用33236，排除7079，技术排除0，耗时245.8s；汇总`/root/autodl-tmp/trd/data/processed/math-v1/logs/20261115-085622-all/summary.json`。


## 2026-11-15：核实用户终端启动的全量数据工程已完成

- 用户询问其他终端启动的一键命令是否跑完。本轮只读核查进程、GPU、manifest及实际产物，未重新运行清洗或分层。
- 两阶段completed均为true；无残留pipeline/vLLM工作进程，两GPU显存均为0MiB。输入40,315，最终可用33,236（82.44%），排除7,079，技术排除0；易4,496／中21,496／难7,244；模型标签30,186、原始标签3,050。运行summary的处理至导出耗时245.77秒（最终同步及退出另计）。
- 核实clean/final的usable/excluded/decisions parquet分别为33,236/7,079/40,315行；实际final/dataset可加载为33,236题，ID唯一，领域非空，难度仅easy/medium/hard。产物data/processed/math-v1/final/dataset，汇总data/processed/math-v1/summary.json；SwanLab https://swanlab.cn/@WangZIyu/trd-math-opd/runs/oqwvc29e 。自动排除理由是模型/规则决策，不在本次状态核查中重新证明清洗精确率。


## 2026-11-15：建议清洗后分层验证集规模

- 用户询问从当前可用数据分层留出的验证集应有多大，强调数量不宜过多。本轮只读统计final/usable.parquet，未真正划分或冻结验证集，无新生成/评分实验。
- 核实33,236题对应33,236个唯一duplicate_group_id；非空参考solution共6,305题（易987/中3,626/难1,692），领域×难度18个组合均有非空参考样本。全体易4,496/中21,496/难7,244。
- 建议固定512题，占全体1.54%，留出后训练32,724题；按难度×领域联合分层，难度比例约易69/中331/难112，稀有领域适当保证覆盖。若用于前述PPL/KL主比较，建议从非空且自动核验通过的参考解答题中选取，并尽量对齐全量的联合分布；应说明其参考解答子分布边界。验证ID及随机种子固定、整个题组与训练隔离；五套最终测试集继续独立。256题可节省成本但统计更粗，1,024题会使每次验证成本约翻倍，因此首选512。当前验证集仍未实际落地。

- 数据分层划分：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/qc1pyqma；配置及日志：`/root/autodl-tmp/trd/preparation/logs/data-engineering/split-20261115-094138`；目标输出：`/root/autodl-tmp/trd/data/processed/math-v1/splits-v1`。
- 划分完成：验证512、训练池32724、首轮训练9984、后备22740，按有效batch128为78步/epoch；零交集检查通过；SwanLab https://swanlab.cn/@WangZIyu/trd-math-opd/runs/qc1pyqma。


## 2026-11-15：冻结512题验证集、训练池和首轮9,984题子集

- 用户确认512题暂定验证集，其余作为可扩充训练池；首轮只训练约1万条，并要求数据工程完整文档与a.md简短索引。选择9,984=78×128，最接近1万；不要求2的幂。已实际执行split.py，SwanLab https://swanlab.cn/@WangZIyu/trd-math-opd/runs/qc1pyqma，没有运行正式生成、训练或验证模型评分。
- 输入为清洗后33,236题。验证512（易69/中331/难112），训练池32,724（易4,427/中21,165/难7,132），首轮9,984（易1,350/中6,457/难2,177），后备22,740。领域×难度联合分层；固定seed20261115；训练池按层内ID哈希顺序和加权缺额交织，首轮取前缀，扩充取更长前缀，因此保留首轮样本。
- 验证参考：6,305非空solution中排除已知pilot来源后6,267候选；快速末boxed/安全等价检查得到2,657条，4次有界Math-Verify终答检查补充稀有层，共2,661条候选。最终512题中508快路径/4符号路径，18层均覆盖。仅核查终答与gold一致，不证明全部推理。原始参考文本不修改；验证参考共234,375 tokens，均值457.76、中位445.5、P95 840.9、最大1,674。未选中的参考检查候选仍留在训练池，不据此重复清洗整题。
- 本地产物：data/processed/math-v1/splits-v1/{validation,train_pool,train_9984,train_reserve}，每份含data.parquet、HF dataset及ID清单；根manifest含来源/代码/产物SHA256、分层计数和SwanLab链接。逐题数据及ID清单按Git规则只留本地；preparation/data_engineering/split-summary.json为小型可提交摘要。
- 自动检查：四个实际HF目录和parquet条数、各文件SHA256均通过；raw_id/sample_id/duplicate_group_id在验证与训练池间交集0，训练子集与后备交集0，全部输入恰好分配；原可用parquet哈希未变；执行脚本SHA256与冻结manifest一致。新增6项划分检查，合计18项CPU回归通过；抽样可重现、跨原始输入顺序稳定、扩充保持子集嵌套。
- 默认入口：新增splits.env.sh；env.sh仅在未显式指定TRAIN_DATA_PATH且冻结manifest存在时启用子集，并保留显式路径，修复run_math_opd.sh再次source环境把用户选定输入重置成原始全集的问题。生成缓存隔离为gen_results/math-v1-splits-v1/train_9984。fresh环境自动激活、显式输入覆盖、shell语法及训练dry-run通过（batch2/acc32、有效128）；训练准备器继续传递split_version/sampling_rank。
- 文档：新增preparation/data_engineering/REPORT.zh-CN.md，详细记录原始数据、64题观察、清洗规则、589对候选判断、训练/测试去污染范围、正式排除数、标签来源和分布、双卡实测、冻结划分/扩充、实际路径及哈希。仓库已有工作约定是Agent.md（AGENTS.md链接至它），已更新其中索引和默认数据范围，并按用户字面文件名新增根a.md短索引；旧PPL/KLD方案与准备README同步到当前划分。验证评分仅准备了输入，尚未接入训练中的自动定期评估。

- PPL/KLD脚本CPU回归验证：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/725t9w03；合成微型Qwen3与临时划分，未使用生产权重或评分真实验证集；日志`/root/autodl-tmp/trd/preparation/logs/ppl-kld-development/20261115-100544-cpu-tests`。

- PPL/KLD脚本CPU回归验证v2：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/deb4gj3t；FP32/BF16合成微型Qwen3、显式CPU LoRA加载及临时划分；日志`/root/autodl-tmp/trd/preparation/logs/ppl-kld-development/20261115-100741-cpu-tests-v2`。

- PPL/KLD脚本最终CPU与合成流程回归：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/c8slwskd；未使用生产权重、未给真实验证集打分；日志`/root/autodl-tmp/trd/preparation/logs/ppl-kld-development/20261115-101333-final-cpu-tests`。

## 2026-11-15：准备固定512题验证集PPL/KLD启动脚本，正式运行交给用户

- 用户要求先阅读Agent.md最新数据划分，再准备好启动脚本由用户自行运行。本轮已按新第6节读取冻结验证512题/训练池32,724/首轮9,984的manifest和实际文件，没有修改划分或启动生产模型的验证评分、生成及训练。
- 新增`preparation/run_ppl_kld.sh`、`preparation/ppl_kld/{config.env.sh,evaluate.py,scoring.py,README.zh-CN.md,test_scoring.py,VALIDATION.json}`。启动器从任意目录运行并自动使用本仓库虚拟环境；默认完整512题、原始4B学生、固定8B教师，可传HF合并模型或LoRA目录并与已有baseline结果比较。配置与旧指标方案、Agent.md索引已同步。
- 固定口径：题目+训练器学生instruction+换行，原始solution单独编码后拼接，不新增EOS/chat/thinking标签；教师学生相同输入和参考前缀，仅统计response。输出学生/教师NLL与PPL、全词表forward KL(T||S)/reverse KL(S||T)、领域/难度/长度分桶；FP32 log-softmax及乘积、FP64归约，不裁剪、不截top-k、不静默丢题或截断。独立于教师额外看到y_o的TRD训练KL。
- 资源差异：评分无梯度/优化器/KV cache，无需训练FSDP2；每卡一个BF16完整师生副本，批次互不重复，默认batch上限8/padded预算8,192/chunk512/FA2/allocator fraction0.92，LoRA保留PEFT FP32推理转换。取隐藏状态后仅对response位置分块投影LM head，避免整段词表logits。实际评分GPU速度/峰值尚未测试，首轮或32题校准由用户启动。
- 校验：真实512题HF目录逐行等于冻结parquet；validation/train_pool的parquet及ID文件SHA256通过，raw_id/sample_id/duplicate_group_id零交集；完整response tokens=234,375，含提示input tokens=281,011，最长input=1,808，默认65个batch。仓库外绝对路径启动、加载env+balanced+splits后32题dry-run、shell语法、Python导入及Git diff检查通过；dry-run不调用评分/CUDA/SwanLab。
- 首轮12项合成回归有1项LoRA加载失败：PEFT自动device_map把微型测试模型从CPU调到CUDA；未使用生产权重或给真实题打分。已显式指定adapter的device_map和torch_device；v2 13项通过，最终16项通过。覆盖首末token/causal shift、HF原生loss与FP64 KL、BF16、chunk/padding一致性、LoRA实际加载、token加权、配对bootstrap、污染/文件篡改拒绝、合成主流程成功/失败及baseline指针、多GPU任务分配模拟。所有软件实验链接见上文，最终SwanLab https://swanlab.cn/@WangZIyu/trd-math-opd/runs/c8slwskd；失败运行也已保留。收尾两卡显存均0MiB。
- 用户运行`bash preparation/run_ppl_kld.sh`后自动在线记录SwanLab项目trd-math-opd并追加HISTORY，输出`preparation/logs/ppl-kld/<label>-<时间戳>-<pid>/`的逐题统计、summary、REPORT及GPU日志。仅完整成功baseline更新latest-baseline.json；后续`--label distilled --student-model <HF或LoRA目录> --compare-to preparation/logs/ppl-kld/latest-baseline.json`核对固定参考/口径/教师文件SHA256后输出前后配对差值和95%区间。没有上传权重、题目数据或token任务，Git仅包含代码、文档、配置和小型检验摘要。

## 2026-11-15 10:18:27 UTC：验证集PPL/KLD baseline

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/348lb6qg；输出/日志：`/root/autodl-tmp/trd/preparation/logs/ppl-kld/baseline-20261115-101823-758028-27412`；学生：`/root/autodl-tmp/trd/models/Qwen3-4B-Instruct-2507`；验证题数512/512。
- 完成：学生PPL=2.14900753，教师PPL=2.04969176，forward KL(T||S)=0.23148751，reverse KL(S||T)=0.21281604 nats/token；有效response tokens=234375；耗时62.9s；汇总`/root/autodl-tmp/trd/preparation/logs/ppl-kld/baseline-20261115-101823-758028-27412/summary.json`。

- 训练/评测脚本CPU合成回归：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/g1ibseb4；日志`/root/autodl-tmp/trd/preparation/logs/training-development/20261115-183237-cpu-tests`；没有启动正式模型训练或benchmark评测。

- 训练/评测脚本CPU合成回归v2：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/zfsssh2l；日志`/root/autodl-tmp/trd/preparation/logs/training-development/20261115-183351-cpu-tests-v2`；没有启动正式模型训练或benchmark评测。

- 训练/评测脚本最终CPU合成回归：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/so9e5dn2；日志`/root/autodl-tmp/trd/preparation/logs/training-development/20261115-183526-final-cpu-tests`；没有启动正式模型训练或benchmark评测。

## 2026-11-15：准备首轮9,984题训练、独立评测和串行总脚本

- 用户要求先读Agent.md，按已划分的9,000多条新数据及优化方法准备训练脚本、评测脚本，以及自动完成训练+测试的总入口；两个阶段独立运行和记录。本轮仅准备脚本、校验冻结数据及CPU合成回归，没有启动正式学生/教师生成、9,984题训练或五个benchmark的模型评测。
- 新入口为 `preparation/run_train.sh`、`preparation/run_eval.sh`、`preparation/run_train_eval.sh`，共享 `preparation/training/config.env.sh` 和 `pipeline.py`。自动加载env、balanced、splits及上游forward_kl_y_r算法默认值：FSDP2/双卡/SP1，batch2/累积32，有效128、1 epoch/78名义更新；训练上限18,432/动态预算32,768；BF16冻结权重、FP32 LoRA、Liger/梯度检查点、全词表KL recompute/chunk512、输出上限16,384，未沿用pilot短输出。
- 训练入口锁定 `data/processed/math-v1/splits-v1/train_9984/dataset`，检查manifest、parquet与ID清单SHA256、实际HF逐行一致，以及训练/验证raw_id、sample_id、duplicate_group_id零交集；拒绝原始全集、MAX_SAMPLES及外部DATA_PATH/CORRECTED_RESPONSES_PATH。生成后再次核对全部9,984题的身份/顺序和非空响应，缺失或过滤丢题则在更新前终止。实际训练集学生提示最长1,669 tokens，0题超过2,048提示预算。
- 独立评测覆盖AIME24/AIME25/HMMT25/BeyondAIME/AMO-Bench，现有题数30/30/30/100/39；pass@16、温度0.6/top_p0.95、TP2，并记avg@16和首8次pass@8。评测保持上游4,096提示预算及38,912输出上限，区别于训练生成16,384；增加评测资源环境参数，当前显存比例0.85、max_num_seqs16、batched tokens43,008。冻结512题PPL/KLD继续使用已有独立入口，未自动混入这两个实验。
- 发现并修正会影响最终准确率的导出问题：原model merger把LoRA拆入lora_adapter子目录，HF主权重仍是基础学生，而原benchmark未加载adapter。新增可选 `--merge-lora`，新训练入口导出时显式启用，使用PEFT CPU safe merge将增量折入最终权重，保留原adapter和lora_merge.json；合并要求非零alpha元数据。新评测拒绝未合入adapter的旧导出，检查所有HF权重分片存在。未改变未启用该选项的上游默认导出行为。
- 每个总运行创建 `<run-id>-train` 与 `<run-id>-eval` 两个独立SwanLab在线实验，以run-id分组。父进程从生成开始统一记录训练配置、控制台及失败，rank0通过relay转发原生训练指标，避免重复实验；评测记录来源训练链接与15个benchmark指标。每阶段summary含成功/失败状态、模型路径及实验链接并追加HISTORY；训练或导出失败不启动评测。单独评测用 `--train-run <成功训练summary.json>` 或显式 `--model`，不按latest/时间排序猜模型；文件锁避免新入口并发抢卡，已有产物拒绝覆盖。
- 产物索引：`preparation/logs/math-runs/<run-id>/{train,eval}/summary.json`及console.log、训练metrics.jsonl、评测results.json/csv与generations；checkpoint/HF导出继续采用既有分盘方案，启动时检查20GiB/18GiB空闲空间。本次收尾系统盘约28GiB、数据盘22GiB空闲，满足现有单实验检查；双卡显存均0MiB。全部原始日志、模型、数据和缓存保持Git忽略。
- 验证：三个入口dry-run、仓库外/tmp绝对路径启动、拒绝原始全集/遗留MAX_SAMPLES=8、shell语法、Python编译、CLI合并选项及Git diff检查通过。首轮17项CPU回归出现5处错误（测试夹具缺少模型源目录；自定义输出根时锁父目录未创建），已修正；v2的17项通过，开发记录驱动误用SDK finish状态值，最终驱动改为默认finish并完成在线上传。最终20项全部通过，覆盖数据污染/篡改、精确模型交接、训练失败停止、独立评测、缺失指标/不完整采样拒绝、真实子进程指标转发，以及随机微型Qwen3的FP32/BF16非零LoRA合并后普通HF加载数值对照；最终SwanLab https://swanlab.cn/@WangZIyu/trd-math-opd/runs/so9e5dn2。没有用生产模型运行这些合成测试。
- 新增 `preparation/training/README.zh-CN.md` 和小型VALIDATION.json；Agent.md第8节及准备README索引已更新。工作分支 `feat/math-train-eval-pipeline`。完整长程GPU训练、实际导出模型和真实benchmark结果须由用户启动正式脚本后验证。
- train 失败：`/root/autodl-tmp/trd/preparation/logs/math-runs/math-trd-9984-v1/train/summary.json`；错误见 `/root/autodl-tmp/trd/preparation/logs/math-runs/math-trd-9984-v1/train/console.log` 和 summary.json。

## 2026-11-15：修复正式训练的SwanLab初始化错误，实际启动后立即停止

- 用户反馈前述命令启动失败，要求定位并实际尝试，确认能启动后立即停止，正式训练与自动评测由用户在另一个终端执行；本次明确不启用实验记录。按本轮用户指示关闭SwanLab/W&B，不创建或上传新的实验；保留这里的代码维护和排查结论。
- 读取 `preparation/logs/math-runs/math-trd-9984-v1/train/summary.json`，失败发生在 `swanlab.init` 读取SDK Settings时，报 `SettingsError: error parsing value for field "project"`，内层为JSONDecodeError。配置导出的 `SWANLAB_PROJECT=trd-math-opd` 被本机SwanLab0.10.5误作嵌套JSON解析；失败早于模型启动，未创建SwanLab实验，实际没有console.log，完整异常在summary.json。此兼容点之前已在数据工程/PPL入口处理，本次正式训练入口漏接，旧dry-run及使用Mock初始化的CPU检查未覆盖。
- 修复 `preparation/training/pipeline.py`：先捕获并暂时移除平面SWANLAB_PROJECT，将项目名作为显式参数传给SDK，finally恢复原环境，训练和独立评测共用这一路径。新增真实SDK Settings解析的回归，用替身阻止创建实验，连续验证train/eval并检查项目名与环境恢复。完整21项CPU回归通过，没有启用SwanLab；本地临时输出 `/tmp/trd-startup-repair-20261115/cpu-regression.log`。
- 实际GPU启动验证加载与正式入口相同的env/balanced/splits和生产辅助函数参数：4B学生、8B教师分别在两卡启动2个TP1生成副本，输入9984题、输出预算16384，学生上下文18432/教师34816、GPU比例0.85、max_num_seqs16/batched tokens34816；约98秒/101秒完成启动。服务全部ready时由临时包装立即退出并关闭Ray，没有提交任何生成请求或写入正式生成缓存。
- 再使用正式torchrun命令构造器和run_training.main初始化双卡FSDP2，师生模型、LoRA、优化器及数据加载器均成功加载；9984个冻结训练ID对应临时初始化fixture，batch2/累积32、名义78步、max_length18432、动态预算32768。临时替换train方法只确认ready并同步退出，未对fixture执行前向、反向或参数更新，未保存checkpoint、未运行评测；约31秒退出，双rank报告global_step=0。该检查覆盖实际模型初始化，不等同于完成真实生成或一次正式更新。
- 临时诊断脚本、逐阶段输出和diagnostic-result.json均在 `/tmp/trd-startup-repair-20261115/`，不加入Git。SDK创建实验的调用在诊断时被替身阻止，但使用真实Settings检查已修复路径；正式默认仍在线分别记录训练与评测。收尾三个实际GPU启动阶段全部returncode0、相关进程已结束，两卡显存均0MiB。
- 原失败run-id目录保留，不覆盖或删除。建议用户直接执行 `bash preparation/run_train_eval.sh`，让入口自动生成新run-id，避免重复使用math-trd-9984-v1触发已有目录保护；正式输入继续锁定新划分9984题，成功后自动合并LoRA并执行五套pass@16评测。

## 2026-11-15T10:58:51.710178+00:00：数学 TRD train / 20261115-185850-279780-53607

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/mym7h666；配置/日志/状态：`/root/autodl-tmp/trd/preparation/logs/math-runs/20261115-185850-279780-53607/train`。

## 2026-11-15T11:10:07.239082+00:00：只读核查正式运行进度并估算耗时

- 用户询问是否约2–3小时完成。本轮只读取运行状态、控制台进度、GPU状态及既有pilot摘要，没有干预正在运行的进程、改参数、启动新实验或评测。当前运行 `20261115-185850-279780-53607`，SwanLab https://swanlab.cn/@WangZIyu/trd-math-opd/runs/mym7h666。
- 当前仍为4B学生y_o生成，完成340/9984，生成计时557秒，累计平均0.610条/秒；线性外推当前阶段尚需约4.39小时，近几分钟速度约对应4–6小时剩余。两卡读取时利用率95%/96%，显存约81139/81145MiB；未见训练torchrun或最终评测进程。
- 估算边界：仅完成约3%的学生生成，已完成短响应会使初期均值偏乐观；tqdm瞬时ETA波动明显，不据此给整套精确结束时间。教师改写、真正FSDP更新及五套pass@16评测尚未实测，无法由当前学生速率可靠外推。结论是2–3小时不足，完整流程宜先按一天左右或更久预留；后续教师阶段有稳定吞吐、更新阶段有单步耗时后再细化。现有pilot主要是短生成和少量压力更新，不能用78个名义更新直接换算正式端到端耗时。

## 2026-11-15T13:45:44.455476+00:00：用户撤回部分数据方案，恢复原9984题完整运行

- 上轮用户希望暂停当前约5754条进度并只蒸馏已生成部分。核实旧生成器在全部完成后才写parquet，磁盘只有prompts，已完成ChatCompletion保存在生成主进程内存；为避免直接终止造成丢失，先SIGSTOP生成客户端54082，再暂停其66个相关进程（含客户端共66个实际恢复进程），尝试只读获取内存但系统ptrace限制不允许；没有改系统ptrace设置，没有终止生成客户端，也未导出或改变数据范围。
- 用户随后明确撤回该方案，要求继续原完整任务。本轮立即SIGCONT恢复66个已暂停进程，但暂停跨越Ray心跳/引擎超时窗口，原两个vLLM后端退出，客户端出现断连与连接重试。为防最后一次重试耗尽而退出丢失内存结果，立即再次只暂停客户端，保留原results列表；此次运行服务中断由Agent的暂停操作触发，已向用户说明。
- 恢复方式：在原HTTP地址172.17.0.6:34035和:38735分别启动独立vLLM服务，模型仍为本地4B学生，TP1×2卡、BF16、max_model_len18432、max_num_seqs16、batched tokens34816、GPU比例0.85、同温度0.6/top_p0.95/top_k20与seed0/1。首个临时包装缺少spawn的main保护导致启动失败（未恢复客户端），补齐保护后两个/health及/v1/models均通过，再SIGCONT原客户端54082。未另起原始生成任务、未重跑已经计入进度的5786条；尚未完成的请求在原异步任务内重试。替代引擎RNG重新初始化，因此不声称与完全不中断运行逐位相同。
- 验证原客户端PID未变、完整任务仍status=running，进度从5786增长至5854并继续至5923/9984，两GPU利用率读取约93%/92%，显存约81019/81061MiB。原pipeline53607及prepare53929继续等待生成结果，输入仍为冻结9984题；后续教师改写、训练和自动评测保持原入口。未执行5000条子集方案、未新建SwanLab实验，沿用 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/mym7h666 。
- 临时恢复服务、状态及日志位于 `preparation/logs/math-runs/20261115-185850-279780-53607/train/recovery/`，均按既有规则忽略Git；supervisor PID74353，服务PID74400/74401，原生成客户端退出后自动停止替代学生服务并释放GPU供原教师阶段使用。恢复摘要recovery-summary.json记录保留进度、服务端口和清理策略；原客户端仍依赖完成全阶段后落盘，当前不能把内存中的轨迹称为已持久化缓存。


## 2026-11-15：统一SwanLab项目复用与实验命名约定

- 用户要求在Agent.md写清：所有实验经过SwanLab；主要目的和阶段相同尽量复用之前project；实验名简洁直白，包含日期、阶段和目的。
- 已扩充Agent.md第4节：覆盖数据/生成/训练/评测/性能/数值校验及失败中断状态；启动前查历史，日期/机器/模型/参数/重跑不构成单独建project理由。当前数学OPD/TRD相关实验复用trd-math-opd，硬件性能专项优先复用trd-pro6000-pilot。
- 统一run名称格式YYYYMMDD-阶段-目的[-关键区别][-rNN]，附清洗分层、TRD9984题、基线PPL/KL和五套pass16评测示例；完整超参留config/description。本轮仅修改约定与历史，未启动实验、改名既有run或干预正在运行的任务。

## 2026-11-15T15:42:03.335385+00:00：只读核实已进入8B教师改写阶段

- 用户只询问当前阶段，明确不要干预任务。本轮只读取进程/GPU/日志及parquet元数据，没有发送信号、修改运行配置或启动新实验。原9984条学生生成已完成并写入math_y_o_responses.parquet，实际9984行；教师改写提示math_y_r_opd_Qwen3-8B_prompts.parquet实际9984行。当前模型进程为Qwen3-8B，输入题目及学生初始解答、生成修正轨迹y_r；最近进度：Generating responses:   0%|          | 23/9984 [01:08<6:16:46,  2.27s/it]，双卡利用率读取95%/94%。
- 尚未启动run_training.py/launch_training.py，当前没有学生参数更新；教师改写完毕才进入FSDP2全词表KL蒸馏，再执行最终评测。新进度条从0计数属于教师阶段，不是重新生成学生轨迹。恢复用的学生服务supervisor已自行退出，状态client-finished，GPU已由原入口教师阶段使用；SwanLab仍沿用 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/mym7h666 。

## 2026-11-15T16:51:25.573173+00:00：评估当前完整运行的磁盘容量与保存峰值

- 用户要求结合剩余空间和后续权重分析磁盘写满中止风险。本轮只读取磁盘、进程、配置、parquet元数据、模型索引及保存/导出源码，没有向后台发送信号、改配置、删文件或启动模型实验；唯一维护性写入为本历史记录。当前仍处于8B教师改写，读取进度1640/9984左右，尚无正式训练checkpoint。
- 实测系统盘overlay总30GiB、剩余约27.48GiB；数据盘/dev/md0总50GiB、剩余约21.30GiB；/dev/shm约110GiB且几乎空闲，inode均有充足余量。checkpoint实际写/root/trd-checkpoints（系统盘）；hf_merged为已存在软链接，指向checkpoints_merged下的数据盘路径，两个st_dev不同。两处checkpoint/导出目录目前仅目录/软链接、尚无大权重文件。
- 基础模型原文件约23GiB已包含在数据盘现有占用，不再计为后续增量。学生safetensors索引total_size=8,045,591,552 bytes（7.493GiB、BF16）；Qwen3层数36、hidden2560、intermediate9728、q_heads32、kv_heads8、head_dim128，LoRA rank64覆盖七种线性层，计算可训练参数132,120,576，FP32 adapter约0.492GiB，Adam两个FP32矩约0.984GiB。按保存源码的分片合计，BF16基础权重+FP32 LoRA+优化器理论约8.97GiB；计绑权重可能重复及元数据约9–10GiB，若意外按FP32保存基础权重则保守按约17–18GiB看待，仍低于系统盘剩余。该体积为源码/配置推算，尚未实测正式checkpoint序列化。
- 已核实训练只保存学生的model/optimizer/extra，没有teacher checkpoint，也没有开启checkpoint目录下额外hf_model完整副本。当前1 epoch/78名义更新、save_steps100，预计只在训练结束保存一次；因此本轮不出现周期保存的旧新两份checkpoint叠加。一般情况下keep1也是新保存成功后才删旧份，并不能把keep1误解为永不出现瞬时双份，但本轮78<100且目录无旧checkpoint。
- HF导出器将FSDP shard在CPU合并、LoRA safe merge后仅写最终HF模型及单独adapter；不会先在数据盘写一套未合并基础权重再写另一整套临时权重。导出器转换为BF16，按可能的embedding/lm_head存储重复保守预计HF+adapter约8–9GiB；再给教师输出、评测轨迹、缓存和日志预留约2–3GiB，数据盘后续预算约10–12GiB，预计仍有约9GiB以上余量。
- 现有生成缓存约113MiB：学生prompts4.52MiB/9984行、学生responses37.58MiB/9984行、教师prompts70.89MiB/9984行。教师responses后续才写入。五个评测共229题×16=3664条响应，上限38912 tokens/条，若按8 bytes/token的文本预算约1.06GiB原始响应文本（是估算假设，不是严格词表字节上界），parquet通常进一步压缩；没有每个benchmark另存一套模型。当前/tmp/ray约139MiB、/root/.cache约88MiB、preparation/logs约28MiB，未见所查主进程持有大体积deleted文件。
- 结论：磁盘写满在保存/导出阶段确实可能使任务失败，但当前单轮配置的剩余容量预计足够，风险较低，无需为容量问题中断任务或立即清理。此判断依赖当前只运行这一套实验且没有额外大文件写入，并非零风险保证；启动器原空间门槛20GiB系统盘/18GiB导出盘当前仍满足。现有实验链接 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/mym7h666；未新增SwanLab实验。
- train 完成：`/root/autodl-tmp/trd/preparation/logs/math-runs/20261115-185850-279780-53607/train/summary.json`；结果：{"model_path": "/root/trd-checkpoints/OPD_MATH/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_20261115-185850-279780-53607/epoch1/hf_merged", "training_data": {"rows": 9984, "sha256": "864efe8abc0a677aecbf78b4dea82158fc7c3479c4d9f6e5580098d0a482def9"}, "output_dir": "/root/autodl-tmp/trd/outputs/OPD/math/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_20261115-185850-279780-53607/epoch1"}。

## 2026-11-16T00:05:19.226495+00:00：数学 TRD eval / 20261115-185850-279780-53607

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/yxzg5pto；配置/日志/状态：`/root/autodl-tmp/trd/preparation/logs/math-runs/20261115-185850-279780-53607/eval`。

## 2026-11-16T01:28:34.210547+00:00：确认训练与模型导出完成，现处于第一套最终评测

- 用户询问后台当前进展及剩余工作。本轮只读取进程、GPU/磁盘、日志、summary和产物元数据，没有向后台发信号、改配置、启动新实验或执行额外评分。
- train/summary.json已completed，训练数据9984条，教师改写结果SHA256=864efe8abc0a677aecbf78b4dea82158fc7c3479c4d9f6e5580098d0a482def9；学生/教师两轮生成均完成，FSDP2完成1 epoch、78个优化器更新，epoch日志参数更新耗时1:37:45。global_step_78完整保存，合并HF模型存在、lora_merge.json标记merged=true。训练阶段wall_seconds=47186.21954170987包含两轮生成等全部准备，不能误作纯参数训练耗时。训练实验 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/mym7h666 。
- eval/summary.json仍running，评测进程run_eval_suite.py，按顺序AIME24/AIME25/HMMT25/BeyondAIME/AMO-Bench、pass@16、TP2。当前generations尚无完成parquet、results.json尚未生成，因此正在第一套AIME24的30题×16次=480条响应；最近进度：Generating responses:  98%|█████████▊| 470/480 [1:18:36<02:57, 17.71s/it]。该约98%仅属于AIME24生成，不代表五套评测/整个实验完成98%。之后仍需判分、其余四套生成/判分及avg@16/pass@8/pass@16汇总、最终SwanLab记录。评测实验 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/yxzg5pto 。
- 已实际落盘容量：FSDP checkpoint约8.971GiB，HF模型与adapter约8.478GiB；实测剩余系统盘约18.30GiB、数据盘约12.46GiB，前一轮容量预算未触发写满。两卡利用率读取95%/96%，后台正常生成评测响应；没有读取或报告尚不存在的最终准确率。固定512题PPL/KLD对比不在当前总脚本内，未把该独立验证当作已经完成。

- AIME评测范围控制CPU回归：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/sqv4bxha；日志`/root/autodl-tmp/trd/preparation/logs/training-development/20261116-095314-aime-scope-checks`；测试使用Mock信号、Mock恢复记录和合成数据，不干预生产进程。

## 2026-11-16T01:59:31.922863+00:00：用户将最终评测缩减为AIME24和AIME25

- 用户确认仅测AIME24/25，要求AIME25结束后不启动HMMT25、BeyondAIME、AMO-Bench。检查AIME24已生成30题×16次、判分写入results.json，pass@16=0.8333333333（25/30）；当前AIME25正在生成。没有重启/暂停当前生成进程，也未修改、隐藏或移动测试集文件。
- 已安装run-local evaluation-control.json，仅选择aime24/aime25。已启动旧五套评测循环的PID120819不能通过改磁盘源文件即时改变内存循环，因此新增判分子进程边界钩子：AIME25的parquet与两套pass@16成绩全部持久化后，核对父PID、创建时间、命令及results_file路径，先关闭判分子进程Ray，再SIGTERM旧评测suite使其finally释放GPU，确保不进入下一测试集；没有使用SIGSTOP。新启动的suite则直接在每套边界读取范围文件并跳过未选集合。
- 旧总控制器仍按五套验证，其受控退出可能暂时写failed/退出异常，这是原范围被用户主动缩减的预期中间状态，不把它冒充五套完成。独立finish_reduced_eval.py监视器等待AIME25边界以及旧控制器退出，汇总仅两套avg@16/pass@8/pass@16，验证各30题且每题16响应，生成CSV，恢复同一SwanLab评测ID yxzg5pto上传六个指标并标completed；summary记录scope_change、明确skipped_datasets三项，并保留原summary作审计，不填伪造三套分数、不新建生产评测实验、不重载模型。
- 当前范围文件 `preparation/logs/math-runs/20261115-185850-279780-53607/eval/evaluation-control.json`；监视状态 `preparation/logs/math-runs/20261115-185850-279780-53607/eval/scope-finalizer-status.json` 为armed、PID142040；监视日志scope-finalizer.log。当前原suite仍PID120819、旧总控制器53607。最近AIME25进度：Generating responses:  35%|███▍      | 167/480 [25:31<1:06:29, 12.75s/it]。范围控制是已布置的后续动作，AIME25尚未完成，未声称结束已发生；评测链接 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/yxzg5pto 。
- 持久默认也调整为AIME24/AIME25：training/config.env.sh与pipeline.py使用MATH_EVAL_DATASETS选择范围，其他集合须显式覆盖；Agent.md默认索引及训练README同步。保持9984题训练数据、模型、采样参数、pass16/TP2及512题独立PPL/KLD入口不变。
- 验证：新增8项CPU范围控制/安全及汇总检查，连同既有训练入口21项共29项通过，包含仅在两套已写成绩后退出、PID重用/其他任务不受影响、先清理与写边界标记再终止、恢复同一记录并只汇总所选六指标、缺失响应拒绝等；信号和Swan恢复使用Mock，没有干预生产进程。SwanLab https://swanlab.cn/@WangZIyu/trd-math-opd/runs/sqv4bxha；日志preparation/logs/training-development/20261116-095314-aime-scope-checks；shell语法、Python编译及git diff检查通过。
- eval 失败：`/root/autodl-tmp/trd/preparation/logs/math-runs/20261115-185850-279780-53607/eval/summary.json`；错误见 `/root/autodl-tmp/trd/preparation/logs/math-runs/20261115-185850-279780-53607/eval/console.log` 和 summary.json。

- 用户缩减后的AIME24/AIME25评测完成：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/yxzg5pto；后续三套未执行；汇总`/root/autodl-tmp/trd/preparation/logs/math-runs/20261115-185850-279780-53607/eval/summary.json`；指标{"eval/aime24/pass@16": 0.8333333333333334, "eval/aime24/avg@16": 0.5583333333333333, "eval/aime24/pass@8": 0.8, "eval/aime25/pass@16": 0.7, "eval/aime25/avg@16": 0.41875, "eval/aime25/pass@8": 0.6}。

## 2026-11-16T03:01:09.230908+00:00：核实AIME24/AIME25评测全部完成并报告成功率

- 用户询问两套是否跑完及成功率。只读核查summary、scope-finalizer状态、results、parquet与进程：最终均completed，完成时间2026-11-16T02:54:40.816345+00:00，原pipeline/suite/finalizer均已退出，两卡显存0MiB、利用率0%；仅有aime24/aime25两份生成文件，各30题、每题16响应、各480条，其他三套没有生成文件。
- 最终蒸馏后学生：AIME24 pass@16=83.333333%（25/30题），平均16次中的单次正确率55.833333%；AIME25 pass@16=70.000000%（21/30题），平均单次正确率41.875000%。首8次pass@8分别80%/60%。pass@16是同题16次中至少一次正确，不将其表述为单次正确率；未新增同口径原始学生benchmark基线，不据此宣称提升。
- 范围控制边界已实际执行：AIME25先完成480条响应与30题判分、写入0.7，然后终止旧五套循环。实际Ray/C++对SIGTERM产生了Aborted/core-dumped退出信息，旧控制器留下中断异常；没有丢失已落盘的两套结果。finalizer随后完成两套CPU汇总，恢复原SwanLab评测ID yxzg5pto，日志确认Upload complete:28 records，最终summary去除中断error并明确记录scope_change及三个skipped_datasets，保留旧summary供审计。最终结果与原控制器退出日志应区分，完成状态以范围变更后的summary和上传结果为准。
- 结果`preparation/logs/math-runs/20261115-185850-279780-53607/eval/results.json`、CSV、summary及scope-finalizer.log均已保留；实验 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/yxzg5pto。本轮没有重新评分、生成或干预任何进程，仅做元数据和响应数量核验，并追加此历史。

## 2026-11-16：定位已完成蒸馏模型并准备PPL/KLD对比入口

- 用户确认训练已完成，要求给出蒸馏后模型的同口径测试脚本，由用户自行运行。本轮核对Agent.md新约定、实际train/summary.json、LoRA合并标记及既有基线；没有启动正式模型评分、生成或训练。
- 定位训练run 20261115-185850-279780-53607：9984题、1 epoch/78更新，status=completed。模型路径由训练summary的result.model_path读取，解析到`checkpoints_merged/OPD_MATH/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_20261115-185850-279780-53607/epoch1/hf_merged`，两个safetensors分片及lora_merge.json merged=true均存在；训练实验 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/mym7h666 。
- 新增`preparation/run_ppl_kld_distilled.sh`：固定这次completed训练记录和完整基线`preparation/logs/ppl-kld/baseline-20261115-101823-758028-27412/summary.json`，自动调用通用评分器，默认双卡、完整512题；输出学生/教师PPL与NLL、全词表forward/reverse KL、分桶和前后配对差值/95%区间，并自动在线SwanLab和追加HISTORY。支持--dry-run，无需用户填写长模型路径。README与Agent.md索引已同步。
- 原基线已完成，学生PPL=2.1490075304，教师PPL=2.0496917593，forward KL=0.2314875093、reverse KL=0.2128160385 nats/token，wall_seconds=62.8691；基线实验 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/348lb6qg 。蒸馏后的指标目前仍未运行，不把训练或AIME结果当作这些指标。
- 启动前按Agent.md最新命名规范调整PPL/KLD实验名：YYYYMMDD-评测-PPL和KL-基线/蒸馏后/校准；同日同目的根据已有本地SwanLab记录递增r02等。继续复用trd-math-opd，不修改评分数学口径或既有基线。
- 检验：仓库外目录执行蒸馏后脚本--dry-run通过，实际512题、234375 response tokens、protocol_sha256与基线一致；shell语法及git diff检查通过。5项CPU合成/命名/主流程相关回归全部通过，在线记录 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/a228n5vk ，日志`preparation/logs/ppl-kld-development/20261116-031549-distilled-launcher-checks`。无生产模型推理，无新增正式蒸馏后评分链接。
- 保留进入会话时已有及其他会话新增的HISTORY、readiness和训练/原始学生评测改动，仅提交本次PPL/KLD入口、相关命名代码/检查、文档与本条历史；未将模型、数据或原始日志纳入Git。

## 2026-11-16 03:19:18 UTC：验证集PPL/KLD distilled

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/iiroiywf；输出/日志：`/root/autodl-tmp/trd/preparation/logs/ppl-kld/distilled-20261116-031913-959867-152167`；学生：`/root/autodl-tmp/trd/checkpoints_merged/OPD_MATH/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_20261115-185850-279780-53607/epoch1/hf_merged`；验证题数512/512。
- 完成：学生PPL=1.82872285，教师PPL=2.04969176，forward KL(T||S)=0.16587333，reverse KL(S||T)=0.18221696 nats/token；有效response tokens=234375；耗时88.9s；汇总`/root/autodl-tmp/trd/preparation/logs/ppl-kld/distilled-20261116-031913-959867-152167/summary.json`。

- Base同口径评测入口CPU回归：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/1v7m2ema；日志`/root/autodl-tmp/trd/preparation/logs/training-development/20261116-112126-base-eval-checks`；合成输入与Mock运行器，未评测生产模型。

## 2026-11-16T03:26:06.836596+00:00：准备蒸馏前4B学生的同口径AIME24/AIME25评测脚本

- 用户要求生成评测脚本，用蒸馏前base模型在相同两个测试集上做相同评测。本轮准备脚本及检查，没有启动原始4B的正式生成/评分，也没有给出尚未计算的base成绩。工作分支feat/math-base-aime-evaluation；保留已有HISTORY追加记录和运行时readiness.json改动。
- 新入口 `preparation/run_base_eval.sh`，执行即可评测原始models/Qwen3-4B-Instruct-2507，不加载adapter、不训练、不调用8B教师；调用base_evaluation.py，复用已成功的benchmark生成/判分链路及独立SwanLab生命周期。仅AIME24/AIME25，强制隔离输出preparation/logs/math-base-eval/<run-id>/eval，模型结果标签<run-id>_base，避免复用蒸馏后缓存或误标distilled。
- 默认参考preparation/logs/math-runs/20261115-185850-279780-53607/eval/summary.json（SwanLab https://swanlab.cn/@WangZIyu/trd-math-opd/runs/yxzg5pto），校验completed、两套各30题及SHA256、最终六指标有效、pass16/温度0.6/top_p0.95/TP2一致；从参考固定提示4096、输出38912、上下文43008、GPU份额0.85、max_num_seqs16、batched tokens43008，显式单机双卡。继承的短输出1024、MATH_EVAL_DATASETS额外集合不能改变本入口的同口径协议。支持--reference-eval或BASE_EVAL_REFERENCE换参考，--run-id/--output-root及--dry-run。
- 实际原始与合并模型generation_config除transformers_version外语义字段相同（含bos/eos/pad、采样默认等），词表/特殊token一致；用两套全部60题逐一检查add_generation_prompt的chat-template编码相同。记录vocab SHA256 c605f1c0a80a9501fca9782b942a7f01474e3f37b1c59e3efea82da0e2f20376、提示token IDs SHA256 74c7034a1550e384df3aad8b2b766374a35952bb5a07c73486bc118d4703e62f；不上传题目或token内容。沿用Rollout单副本replica_rank0加默认seed0，没有新增RolloutConfig不支持的seed字段或修改生产生成器。
- 扩展共享evaluation可区分base/distilled标签、记录参考来源，并在基线评测完成时生成comparison.json/csv：base、蒸馏后以及“蒸馏后减base”的百分点差，同时上传比较指标。原蒸馏评测文件不改写。共享experiment可传明确名称，基线复用trd-math-opd，名称YYYYMMDD-评测-AIME24和25-pass16-基线，同日重跑根据本地记录自动r02等，遵循Agent.md命名约定。
- 验证：真实root目录dry-run通过；/tmp下绝对脚本路径启动，并故意继承EVAL_RESPONSE_LENGTH=1024、额外hmmt25集合，仍固定为38912和AIME两套，dry-run不创建实验目录、未启动GPU或SwanLab实验。新增6项CPU回归连同既有29项共35项全部通过，覆盖参考哈希/参数/成绩异常拒绝、60题输入差异拒绝、adapter模型拒绝、dry-run无评分、规范命名重跑和base标签独立/百分点对比，未评测生产模型。SwanLab数值校验 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/1v7m2ema；日志preparation/logs/training-development/20261116-112126-base-eval-checks。shell语法、Python编译及git diff检查通过，双卡显存均0MiB。
- 使用说明补充至preparation/training/README.zh-CN.md、准备README及Agent.md第8节；用户正式运行命令为bash preparation/run_base_eval.sh。正式结束后才会出现base准确率、SwanLab基线评测链接和前后比较结果，现有蒸馏结果83.33%/70%的pass16不作为base值。

## 2026-11-16：完成蒸馏前后PPL/KLD对比与初步分析报告

- 用户提供蒸馏后运行完成日志，要求与蒸馏前指标比较、初步分析并形成报告。本轮读取baseline-20261115-101823-758028-27412和distilled-20261116-031913-959867-152167的summary、逐题统计及训练记录；没有重新生成、训练或运行模型评分。
- 可比性复核：两份逐题文件SHA256通过；512个sample/raw/group ID、参考前缀token哈希、prompt/response token数及难度/领域均配对一致；评分协议与教师内容哈希分别相同。独立复算全局及所有分桶与原summary一致；有效234375 response tokens，教师总体NLL/PPL相同，逐题teacher NLL总和最大差约4.55e-13。
- 主要变化：学生PPL 2.1490075304→1.8287228475（-14.90%）；NLL 0.7650061218→0.6036178257（-21.10%）；forward KL(T||S) 0.2314875093→0.1658733280（-28.34%）；reverse KL(S||T) 0.2128160385→0.1822169638（-14.38%）；教师PPL均2.0496917593。原评分生成的1000次按题配对bootstrap区间均位于0以下；PPL区间[-0.343688,-0.299250]，FKL[-0.070461,-0.061122]，RKL[-0.034809,-0.026472]。
- 分布与边界：512题NLL/PPL全部下降；FKL 496题下降/16题上升，RKL 344题下降/168题上升。易中难与各领域的总体指标均改善，但<=256参考tokens的111题RKL 0.3459868285→0.3605456053（+4.21%）。难题PPL降18.24%，残余PPL和KL仍最大。微积分5题、其他3题以及>1024 tokens仅3题，不作稳定子领域/长推理推断；学生参考PPL低于教师不表示整体超过教师，也不直接表示数学正确率提升，bootstrap不覆盖训练种子或系统偏差。
- 报告产物：`preparation/ppl_kld/reports/20261116-comparison.zh-CN.md`，附同名前缀JSON数值摘要、CSV及SVG三指标图（合计约47KB）；图已人工可视检查、链接与主要数值断言通过、git diff检查通过。PPL/KLD README增加报告索引；原始两轮summary/REPORT与逐题文件均未覆盖。
- 原实验：基线 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/348lb6qg ，蒸馏后 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/iiroiywf ，训练 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/mym7h666 。本次CPU统计复核/作图与报告继续复用trd-math-opd，命名20261116-评测-PPL和KL对比报告，链接 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/qqo1vjmx；生成脚本快照、作图PNG及SwanLab日志保存在`preparation/logs/ppl-kld-analysis/20261116-033413`，最终文字在同一analysis ID恢复上传。没有上传权重、题目或逐题原始数据。
- 本轮仅提交报告/小型汇总/图表、README索引和本条历史，保留已有运行追加历史及readiness等其他改动。


## 2026-11-16T06:35:52.056770+00:00：核查首轮AIME退化原因与训练监督

- 用户询问蒸馏后两个数学测试集结果下降的原因。本轮读取既有base/蒸馏后summary、四份AIME生成parquet、9984条学生初稿/教师改写、78步训练指标、训练/生成源码、固定参考PPL/KLD报告及原论文/官方模型卡；只做CPU终答重新判分、分词、配对统计和报告，没有重新生成、加载生产模型或训练。SwanLab继续复用trd-math-opd，实验名20261116-评测-AIME下降原因核查：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/z5giapo2。
- 重新判分与原summary一致，逐题prompt/gold配对相同。AIME24正确响应289/480→268/480（avg下降4.375个百分点），pass16同为25/30且覆盖同一25题；AIME25为221/480→201/480（avg下降4.1667个百分点），pass16为23/30→21/30，实际新增1题/丢失3题。合并60题平均正确率下降4.2708个百分点；按题配对20000次bootstrap区间[-7.1875,-1.5625]个百分点，seed20261116。AIME25覆盖差的精确McNemar p=0.625，不能把覆盖少两题单独解释为稳定退化；bootstrap不包含重训练/重生成种子差异。
- 实现风险：KLTrainingDataset._build_sequence直接拼纯文本prompt+换行+response，不使用chat template或显式追加EOS；生成/评测走chat/completions。该差异已存在于初始提交21116fb，非本机优化新增；固定参考PPL/KLD也采用纯文本，因此已有改善没有验证chat输入下独立生成。作为优先控制变量项，不宣布已证明因果。
- 监督核查：全部9984条教师改写均以think开始，9949闭合/35未闭合；教师思考段占目标token70.7776%，5004条思考段包含rewrit。学生是non-thinking 4B-Instruct-2507，教师8B默认thinking；实际AIME输出前后均无think标签，不能说评测时学生已变成thinking模型。学生不可见初稿上的教师改写思考及模式差异可能影响监督，70.78%是token份额，非测得的梯度份额。
- 同一boxed判分器下训练初稿通过7153/9984（71.6446%）→改写7310/9984（73.2171%），206正向/49反向变化/2625两者未通过，训练消费全部目标未按此次核验筛选。整数gold5829题通过4485→4591（76.9429%→78.7614%），129正向/23反向变化。未通过不等同于严格证明错误，字符串判等不能替代推理核验；不能假定教师改写一定正确。
- 其他证据：无可提取boxed响应前后总计均38/960，不能用总量格式缺失解释下降；重新编码长度接近38912上限的响应95→118，但两套均长和中位数下降，未保存finish_reason，不能把全部退化归因于截断/整体变长。随机512条训练长度复核中13条裁教师初稿提示块，0条裁目标尾部；78步日志数值均有限，无发散证据。只有最终模型，无法区分欠训练、过拟合或能力干扰。
- 分析结论：本轮未提升独立解题；PPL/KL降低表示固定参考前缀上预测改善，不等于长程自由生成改善。当前优先检查chat格式、教师thinking/改写思考和目标质量；主因需控制变量实验，不能直接断言教师弱、学习率大、训练太少或硬件优化有错。后续建议先对齐协议并用独立验证的自由生成指标和中间checkpoint做小规模对照，再决定扩大训练。
- 产物：preparation/training/reports/20261116-aime-regression.zh-CN.md与同名前缀JSON，训练README增加索引。CPU脚本快照、逐题正确次数/长度、原文件及源码哈希、console/SwanLab日志保存在preparation/logs/aime-regression-analysis/20261116-review/；原始响应和生产代码未覆盖。全部本地报告链接、保存指标断言及git diff --check通过；同一SwanLab分析ID恢复上传整数子集核查和最终报告。Git只提交报告/小摘要/索引和本条历史，保留会话前已有HISTORY追加与readiness运行改动。


## 2026-11-16T06:39:30.563143+00:00：确认修复阶段缓存复用边界与实施顺序

- 用户询问接下来准备怎样修复，以及是否可以复用既有学生/教师轨迹。本轮复查真实缓存字段、KLTrainingDataset、训练启动器、training/pipeline.py缓存/输入校验及既有诊断；未改训练代码或启动新生成、训练、评测实验。
- 结论：首次格式修复及目标筛选/派生对照可直接复用原9984条学生初稿y_o、教师改写y_r及其中initial_response，不必全量重新采集；修复后KL所需的冻结教师logits仍须在新上下文中重新前向计算，不等于重新生成文本。若更换初稿学生、教师或采样提示/模式，并要求新策略生成轨迹，则相应阶段才需重采集。
- 对齐对象：师生保留各自合法提示（教师额外看到初稿属于refine OPD设计），在同一目标y_r上对齐response token与因果预测位置；不是把y_o和y_r逐字对齐，也不是强制两边看同一prompt。移除教师think监督要同步处理教师评分上下文与loss mask，不能仅删标签；完整输出的结束标记与疑似截断样本分别处理。
- 计划顺序：先准备显式复用旧缓存且校验来源/样本ID/哈希的只训练入口，输出隔离；修复chat模板、response边界、mask和结束标记并做token级及小批次数值检查；设置格式修复单变量对照，再分别比较去除教师改写think监督与可靠终答筛选。符号答案不用单纯字符串失败直接判错；保留无法确认者单独统计。各方案从原始4B重新初始化LoRA，冻结8B，并保持数据划分及可比超参。
- 验证：优先固定独立验证集自由生成正确率/长度/结束情况并保留中间checkpoint，PPL/KL保留旧协议供历史对比，新增chat协议单独记录；选择方案后再正式训练及AIME24/25同口径最终评测。所有实际实验继续复用SwanLab项目trd-math-opd。既有分析实验https://swanlab.cn/@WangZIyu/trd-math-opd/runs/z5giapo2，本轮只读核查和方案记录不新建实验。

## 2026-11-16T06:57:58.564553+00:00：数学 TRD 数值校验 / chat-repair-cpu-20261116-065758

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/mfo48l7h；配置/日志/状态：`/root/autodl-tmp/trd/preparation/logs/training-development/chat-repair-cpu-20261116-065758`。
- 数值校验 失败：`/root/autodl-tmp/trd/preparation/logs/training-development/chat-repair-cpu-20261116-065758/summary.json`；错误见 `/root/autodl-tmp/trd/preparation/logs/training-development/chat-repair-cpu-20261116-065758/console.log` 和 summary.json。

## 2026-11-16T06:59:21.326061+00:00：数学 TRD 数值校验 / chat-repair-cpu-20261116-065921

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/l3darndn；配置/日志/状态：`/root/autodl-tmp/trd/preparation/logs/training-development/chat-repair-cpu-20261116-065921`。
- 数值校验 失败：`/root/autodl-tmp/trd/preparation/logs/training-development/chat-repair-cpu-20261116-065921/summary.json`；错误见 `/root/autodl-tmp/trd/preparation/logs/training-development/chat-repair-cpu-20261116-065921/console.log` 和 summary.json。

## 2026-11-16T07:00:53.145353+00:00：数学 TRD 数值校验 / chat-repair-cpu-20261116-070052

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/rcywdvtl；配置/日志/状态：`/root/autodl-tmp/trd/preparation/logs/training-development/chat-repair-cpu-20261116-070052`。
- 数值校验 完成：`/root/autodl-tmp/trd/preparation/logs/training-development/chat-repair-cpu-20261116-070052/summary.json`；结果：{"sample_rows": 512, "seed": 20261116, "aligned_causal_targets": true, "eos_rows_in_audit": 511, "target_truncated_rows_in_audit": 0, "cache": {"source_summary": "/root/autodl-tmp/trd/preparation/logs/math-runs/20261115-185850-279780-53607/train/summary.json", "source_summary_sha256": "65f8ed28266e865fa50d1796d97fb084122940e59a057d4efebe3e1dc1ba8f77", "source_swanlab": "https://swanlab.cn/@WangZIyu/trd-math-opd/runs/mym7h666", "teacher_cache": "/root/autodl-tmp/trd/gen_results/math-v1-splits-v1/train_9984/OPD/math/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_20261115-185850-279780-53607/math_y_r_opd_Qwen3-8B_responses.parquet", "teacher_cache_sha256": "864efe8abc0a677aecbf78b4dea82158fc7c3479c4d9f6e5580098d0a482def9", "student_cache": "/root/autodl-tmp/trd/gen_results/math-v1-splits-v1/train_9984/OPD/math/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_20261115-185850-279780-53607/math_y_o_responses.parquet", "student_cache_sha256": "88dda41794ad55ef4729aa8e2080ed5984d3170246b80457122ac905c96514f7", "rows": 9984, "generation_response_limit": 16384, "new_generation": false, "derived_cache": "/root/autodl-tmp/trd/preparation/logs/training-development/chat-repair-cpu-20261116-070052/cache/chat-training.parquet", "derived_cache_sha256": "55af3f7c9a658074b8004f545aa31f6a85d07b12f5cdc97328fbd6cd03ad5fe6", "sequence_format": "chat", "target_text_changed": false, "filtered_rows": 0, "complete_rows_with_eos": 9932, "completion_basis_counts": {"inferred_complete": 9932, "near_generation_limit": 44, "no_parseable_box": 2, "unclosed_thought": 6}, "completion_caveat": "Inferred from saved text; original finish_reason was not stored."}, "new_generation": false, "new_model_inference": false}。

## 2026-11-16T07:03:05.121022+00:00：数学 TRD 数值校验 / chat-repair-gpu-20261116-070304

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/y86py49p；配置/日志/状态：`/root/autodl-tmp/trd/preparation/logs/training-development/chat-repair-gpu-20261116-070304`。
- 数值校验 完成：`/root/autodl-tmp/trd/preparation/logs/training-development/chat-repair-gpu-20261116-070304/summary.json`；结果：{"optimizer_steps": 2, "microbatches_per_rank": 64, "sample_visits": 256, "all_values_finite": true, "gpu_peak_gib": [81.54217529296875, 57.43865966796875], "wall_seconds": 201.169836435467, "cache_manifest": {"source_summary": "/root/autodl-tmp/trd/preparation/logs/math-runs/20261115-185850-279780-53607/train/summary.json", "source_summary_sha256": "65f8ed28266e865fa50d1796d97fb084122940e59a057d4efebe3e1dc1ba8f77", "source_swanlab": "https://swanlab.cn/@WangZIyu/trd-math-opd/runs/mym7h666", "teacher_cache": "/root/autodl-tmp/trd/gen_results/math-v1-splits-v1/train_9984/OPD/math/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_20261115-185850-279780-53607/math_y_r_opd_Qwen3-8B_responses.parquet", "teacher_cache_sha256": "864efe8abc0a677aecbf78b4dea82158fc7c3479c4d9f6e5580098d0a482def9", "student_cache": "/root/autodl-tmp/trd/gen_results/math-v1-splits-v1/train_9984/OPD/math/Qwen3-4B-Instruct-2507/teacherQwen3-8B_y_r_kl_forward_full_vocab_clip0_refine_20261115-185850-279780-53607/math_y_o_responses.parquet", "student_cache_sha256": "88dda41794ad55ef4729aa8e2080ed5984d3170246b80457122ac905c96514f7", "rows": 9984, "generation_response_limit": 16384, "new_generation": false, "derived_cache": "/root/autodl-tmp/trd/preparation/logs/training-development/chat-repair-cpu-20261116-070052/cache/chat-training.parquet", "derived_cache_sha256": "55af3f7c9a658074b8004f545aa31f6a85d07b12f5cdc97328fbd6cd03ad5fe6", "sequence_format": "chat", "target_text_changed": false, "filtered_rows": 0, "complete_rows_with_eos": 9932, "completion_basis_counts": {"inferred_complete": 9932, "near_generation_limit": 44, "no_parseable_box": 2, "unclosed_thought": 6}, "completion_caveat": "Inferred from saved text; original finish_reason was not stored."}, "weights_saved": false, "new_generation": false}。

## 2026-11-16T07:06:59.717082+00:00：数学 TRD 数值校验 / chat-repair-final-20261116-070659

- SwanLab：https://swanlab.cn/@WangZIyu/trd-math-opd/runs/bf3w6lhm；配置/日志/状态：`/root/autodl-tmp/trd/preparation/logs/training-development/chat-repair-final-20261116-070659`。
- 数值校验 完成：`/root/autodl-tmp/trd/preparation/logs/training-development/chat-repair-final-20261116-070659/summary.json`；结果：{"regression_tests": 47, "shell_syntax": "passed", "train_eval_dry_run": "passed", "new_production_model_inference": false}。


## 2026-11-16T07:12:40.308217+00:00：实现缓存复用及聊天格式修复，完成CPU与双卡校验

- 用户确认开始执行修复方案。工作分支fix/cached-chat-distillation-20261116；先实施格式修复对照A：沿用原始9984题及所有教师目标文本/think，不筛题、不改主要优化参数，原始4B重新初始化LoRA、冻结8B。没有重新采集学生或教师轨迹；后续think监督及可靠终答筛选分别控制变量。
- 新入口preparation/run_repair_train.sh和run_repair_train_eval.sh，固定引用首轮completed train summary；校验冻结划分、模型路径、教师缓存原SHA256、学生/教师ID及gold、初稿与initial_response和实际教师提示内容。缓存缺失或不匹配拒绝，不调用prepare_training_data.py。派生train/cache/chat-training.parquet保留全部9984行/顺序/response，manifest记录来源和哈希。
- 新增sequence_format=chat可选训练模式；师生分别使用各自tokenizer、原始生成user提示和assistant起始标记；保留teacher额外初稿。response直接编码以避免教师模板改写think空白，显式校验词表/结束token，按同一response和causal预测位置对齐。旧plain默认及原PPL/KLD协议保留。
- 旧缓存没有finish_reason，结束状态保守推断：9932条加EOS，44条接近上限/6条非上限未闭合think/2条其他无法解析boxed不加EOS；这52条仍训练现有response，不按正确率筛选。完整并不证明答案正确，原始响应文本不变。
- CPU校验SwanLab https://swanlab.cn/@WangZIyu/trd-math-opd/runs/rcywdvtl ：46项通过，完整缓存派生并随机512题逐token检查师生因果目标完全一致、0目标尾部裁剪。日志preparation/logs/training-development/chat-repair-cpu-20261116-070052。首次夹具Mock URL不能JSON序列化 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/mfo48l7h ，第二次夹具缺reward_model https://swanlab.cn/@WangZIyu/trd-math-opd/runs/l3darndn ，均在测试阶段失败、无生产GPU训练，失败状态和日志保留。
- 原始4B/8B双卡FSDP2短程校验 https://swanlab.cn/@WangZIyu/trd-math-opd/runs/y86py49p ：balanced batch2/acc32、18K上限/32K动态预算、BF16/FP32 LoRA、recompute KL、FA2/Liger/重分片；64微批/rank、2次优化器更新、256次样本访问，两rank成功且已记录数值有限。NVML峰值81.5422/57.4387 GiB，耗时201.17秒；不保存短程权重。日志preparation/logs/training-development/chat-repair-gpu-20261116-070304，保留rank指标、配置、源码快照和显存采样。
- 最终CPU回归及dry-run https://swanlab.cn/@WangZIyu/trd-math-opd/runs/bf3w6lhm ：47项通过，包含因果shift与HF原生NLL、原提示/目标保真、EOS与不完整样本、截断对齐、旧plain不变、来源/词表/模型/划分拒绝及不启动生成器等；正式train+eval dry-run确认9984题/78步/仅AIME24和25。源码AST、shell语法、文档链接及Git diff检查通过。
- 保留旧模型和checkpoint，不删权重。当前两盘剩余约18/12GiB；通用空间门槛20/18GiB保留，新修复入口仅在已实测BF16/4B/LoRA64/有效128/1epoch/78步且不周期保存全量状态时，以12/11GiB门槛分别保留余量（旧实测checkpoint8.971/导出8.478GiB），配置不符拒绝该门槛。该对照仅保存最终checkpoint，中途checkpoint选择仍属后续验证工作。
- 新增修复说明preparation/training/REPAIR.zh-CN.md、小型repair-validation.json，更新训练README和Agent.md索引。Git仅提交代码、配置、文档及小摘要；原始模型、缓存、逐题数据、日志与原有readiness改动均不纳入本次提交。下一步启动repair-chat-9984-v1，格式修复是否改善准确率仍待完整训练及独立评测。
