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
