# PROJECT_STATUS：Ascend NPU 训练性能优化交接

> 更新时间：2026-08-14  
> 本文只记录已核验事实、采用中的代码和下一阶段可执行事项。性能原始日志与 profiling 数据不在本地保存。

## 1. 当前目标

- 在不改变模型最终功能、训练语义以及 `loss`/梯度可接受性的前提下，优化 8 卡 Ascend NPU 训练性能。
- 固定主验收合同：客户配置、`batch_size=16/rank`、8 rank、后 8 张 NPU、完整名称为 `mapqr-leicheng` 的既有容器。
- 最终目标：同合同下 8 卡 NPU 与 8 卡 GPU 的 `throughput (samples/s)` 达到 `1:1` 或更好。
- 永久算法基线固定为 `63861df 【loss对齐】随机性移除`；客户字段对齐后的基线工作树为 `codex/baseline-customer-runtime-config@4c37039`。累计收益不得改用最新提交作为基线。
- 每个通过验证的算子或完整功能只形成一个可独立回退的 commit，提交信息使用 `【npu性能优化】<对象与动作>`。

## 2. 当前代码和资源状态

- 权威优化分支：`ascend_npu_optimize`。
- 当前采用版本：`f922c389725574257f177c14ff34dda51c6c5c67`，提交标题为 `【npu性能优化】MSDA切换DrivingSDK融合实现`。
- 当前直接父提交：`bf9ed6e`。
- 基线分支：`codex/baseline-customer-runtime-config@4c37039`。
- 远端权威仓库工作区已核验为 clean；保留上述优化工作树和客户字段基线工作树。
- `mapqr-leicheng` 容器存在且当前训练进程为 0；本阶段结束时不占用 NPU。
- STEP-192 的远端诊断目录、脚本和本地一次性脚本/字节码均已删除，匹配数量为 0。
- 本地迁移记录仓库存在大量规划文档、客户输入和历史材料的未提交变更/未跟踪文件；它不是权威训练代码仓库，不应整体清理或整体提交。

## 3. 已采用并合入的优化

从永久基线 `63861df` 到当前 `f922c38` 的提交链如下：

| Commit | 功能 | 主要文件 |
|---|---|---|
| `fb979b2` | SOAP 预条件器改为 NPU 亲和路径 | `projects/mmdet3d_plugin/optimizers/soap.py`、客户训练配置 |
| `6477a5b` | DataLoader 多进程加载 | 客户训练配置 |
| `5a37d04` | 移除训练输入哈希调试开销 | `projects/mmdet3d_plugin/models/detectors/spetr3d.py` |
| `14d4f23` | SOAP 分块 Foreach 调度 | `projects/mmdet3d_plugin/optimizers/soap.py` |
| `a757f29` | 客户训练配置字段对齐；非性能提交 | 客户训练配置 |
| `b36821e` | GeometricLoss 有限值索引消除 | `projects/mmdet3d_plugin/models/losses/geo_loss.py` |
| `bf9ed6e` | TextLogger 显存统计同步降频 | `mmcv/runner/hooks/logger/text.py`、客户训练配置 |
| `f922c38` | MSDA 切换到 DrivingSDK 融合实现 | `projects/mmdet3d_plugin/models/utils/multi_scale_deformable_attn_function.py` |

配置文件为：

`projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py`

DrivingSDK 计划中的可达性检查、语义门禁、8 卡 A/B 和重新 profiling 已执行。当前队列中实际可用且通过完整验收的是 MSDA；其余候选不能因为 API 存在就直接替换。

## 4. 关键性能与正确性结果

### 4.1 永久基线派生版与当前版本：相同 30-step 口径

| Metric | `4c37039` | `f922c38` | Change |
|---|---:|---:|---:|
| `time` mean | 37.440000 s | 9.269933 s | -75.241% |
| `throughput (samples/s)` | 3.419000 | 13.808082 | +303.863% |
| `memory` max | 28460 MiB | 26848 MiB | -5.664% |

说明：`4c37039` 是从永久算法基线派生出的客户字段对齐版本，用于相同客户运行字段的 30-step 累计性能比较；永久算法锚点仍是 `63861df`。

### 4.2 DrivingSDK MSDA 的直接增量：相同 876-step 长训

| Metric | Parent `bf9ed6e` | Current `f922c38` | Change |
|---|---:|---:|---:|
| `time` mean | 8.602508 s | 8.165771 s | -5.077% |
| `time` median | 6.436000 s | 6.050000 s | -5.998% |
| `time` P95 | 25.846000 s | 25.248000 s | -2.314% |
| `throughput (samples/s)` | 14.879382 | 15.675189 | +5.348% |
| `memory` max | 27173 MiB | 27175 MiB | +2 MiB |

- 两版均完成 876/876 step，四个 epoch 的 `loss` 和 `grad_norm` 全部有限。
- 最终 epoch 的 `loss` mean 差异为 +0.155%，下降趋势一致。
- `iter_876.pth` 保存和从该 checkpoint 恢复到 881 step 均通过。
- 同 checkpoint、512-sample 推理中，`task/s` 从 14.2 提升到 15.8（+11.2676%），`elapsed` 从 36 s 降到 32 s（-11.1111%）；固定 shape 输出无 nonfinite mismatch，已记录的 NRMSE 约为 `1.93e-4` 和 `1.11e-4`。

### 4.3 GPU 主参照：最大公共窗口 1～876 step

| Metric | GPU 8×A800 | NPU `f922c38` | Gap |
|---|---:|---:|---:|
| `time` mean | 4.515542 s | 8.165771 s | NPU 慢 1.808× |
| `time` median | 4.312500 s | 6.050000 s | NPU +1.737500 s |
| `time` P95 | 5.543500 s | 25.248000 s | NPU +19.704500 s |
| `throughput (samples/s)` | 28.346540 | 15.675189 | NPU:GPU ≈ 0.553:1 |
| `memory` max | 28816 MiB | 27175 MiB | NPU 少 1641 MiB |

边界：GPU 日志与 NPU 训练的数据量和随机语义并非严格一致；GPU step1 曾因 NaN 跳过更新、step2/3 的 `grad_norm` 为 `inf`。因此该表用于性能主参照，功能正确性由 NPU 同合同父/当前长训、resume 和同 checkpoint 推理 A/B 承担。

## 5. 最新稳定步 profiling 结论

采集版本为 `f922c38`，客户 `batch_size=16/rank`、后 8 NPU、8 rank。只 profile 预热后的普通 Step7；原始 profiling 已在远端原位分析后删除。

- `service_ms=7762.3855`
- `device_busy_union_ms=1916.08275`
- `underfeed_ratio=75.3158%`
- 243 个唯一算子、84,811 次 device kernel 调用，kernel duration sum=`1928.575394 ms`
- HCCL kernel sum 仅 `12.637526 ms`，且大部分与计算重叠，不是当前 P0。

按单算子 Top N：

| Priority | Operator | Device Time | 状态 |
|---:|---|---:|---|
| 1 | `Conv2D` | 271.631355 ms | 待分析冻结路径 `Conv+BN` 折叠 |
| 2 | `MSDA Grad` | 187.430755 ms | DrivingSDK 已合入并通过长训 |
| 3 | `Conv TransData` | 102.938092 ms | 直接 channels-last 方案不受当前环境支持 |
| 4 | `BatchMatMulV2` | 82.961842 ms | 待定位具体 Attention 调用点 |
| 5 | `MSDA forward` | 82.345215 ms | DrivingSDK 已合入并通过长训 |

按调用链聚合：`Conv backbone=490.566843 ms`、`MSDA=269.775970 ms`、`Layout/Copy=243.360041 ms`、`Attention/MatMul=224.836641 ms`、`Elementwise=213.197111 ms`、`Index/Reduction=198.300161 ms`。

完整脱敏清单保存在本地：

- `STEP-189_f922c38_全部算子耗时.csv`
- `STEP-189_f922c38_全部算子耗时.md`
- `STEP-189_f922c38_算子类别耗时.csv`

Top N 不能机械地把核心 `Conv2D` 删除或替换；必须把算子、辅助 TransData、前后算子和活跃源码映射到同一调用链，计算可消除的独占上限后再做单变量门禁。但下一阶段的候选顺序必须继续由上述 Top N 驱动。

## 6. 关键技术决策

- 性能累计比较永久锚定 `63861df`；客户字段对齐后的可运行基线使用 `4c37039`，不能滚动改成最新提交。
- 正式训练、基线、A/B 和 profiler 只能在 `mapqr-leicheng` 中用 8 张 Ascend NPU；固定驱动、CANN、PyTorch 2.7.1、torch_npu 2.7.1 和既有依赖版本。
- profiler 采用“先让训练稳定，再采集一个代表性 step”；SOAP 周期重步与普通步分开归因。最终性能比较使用完整 30 step 或最大公共窗口，不只报告普通步。
- 原始 profiler 只在远端诊断目录临时保留；分析和脱敏摘要生成后立即删除。
- 候选流程固定为：Top N/调用链归因 → 理论可消除上限 → 真实 shape 机制与数值门禁 → 8 卡 30-step A/B → 必要时长训/resume → 单功能 commit。
- 当前目标是算子替换、融合或向量化。布局和运行时开关可以用于解释热点，但不应冒充算子优化成果。
- 远端现有环境不得安装、升级或替换组件；本地缺少分析工具时可安装并记录。

## 7. 已知问题

- 当前 NPU:GPU 条件吞吐约为 `0.553:1`，离 1:1 仍需约 +80.837% 吞吐，或平均每步减少约 3.650229 s。
- 普通步 device underfeed 为 75.3158%。现有 Level0 证据将主要空泡标记为 `possible_host_launch_lag`，但没有足够 source stack 证明单一 Python 根因；不能据此直接修改业务代码。
- SOAP 周期 QR 仍是周期重步最大单点：`aclnnLinalgQr_QrAiCPU_Qr=22.641383956 s`，占该重步 device busy 94.721%。当前固定环境中没有验证通过且逐状态等价的更快实现。
- 最终完整任务评测尚未闭环：当前 test dataset 配置中的 `lidar_type` 与活动 dataset 构造签名不兼容；即使诊断性内存移除该字段，远端对象存储/标注在限时内也未证明可达。客户既有绝对 `F1/Precision/Recall/IoUMean` 不能绑定到本轮 checkpoint。
- 远端评测需要的既有 OR-Tools 环境未在固定训练容器中满足；禁止在远端临时安装或用 shim 冒充正式评测。
- 本地 GPU 配置与当前 NPU 生效配置并非同一文件 SHA；需要继续逐字段分类硬件适配、已批准性能差异和非合同差异，不能直接宣称严格同合同。

## 8. 已尝试但失败或已拒绝的方案

- `channels_last`：在固定 torch_npu 2.7.1 上调用 NPU memory format 直接报错：仅支持 `contiguous_format` 或 `preserve_format`。失败发生在算子计时前，没有性能数据、没有业务修改；备用 permute 绕行不属于算子替换且会引入额外转换，已放弃并清理。
- MatMul HF32：代表 shape 预计仅约 `3.964 ms/step` 收益，且最坏 NRMSE=`1.469314e-4`，低于阶段门槛，未进入训练 A/B。
- Inplace ReLU 改为 out-of-place：现有耗时由 9.26 倍元素量解释，in-place 单位元素反而快 19.33%；最大激活会新增约 4.23 GB 输出，已拒绝。
- Conv HF32：完整 30-step 回归，已关闭。
- 全局 internal format、TASK_QUEUE、COMBINED、CPU affinity、无条件 DataContainer pin：正式门禁无收益或回归，已关闭。
- Point sampling packed BMM、PointPillarScatter 向量化：微基准有效但 30-step 回归，已关闭。
- PillarVFE layout、SECOND stack+sum、显式 grad-clip foreach：收益低于门槛或框架已自动采用有效路径，已关闭。
- MapTR Nonzero/Unique/IndexPut、标准 MHA、BEV Pool 等 DrivingSDK 候选：已按活跃性、语义和 A/B 证据关闭，不因新 profile 出现同名算子而重复测试。
- SOAP QR batching/out-buffer/multi-stream、固定环境替代实现：无逐状态等价且稳定的收益方案；分块外部实现还要求改变既有版本或算法语义，不采用。
- SOAP QR 项目内自定义算子审计：固定环境虽有 NpuExtension、AscendC/TBE 编译工具链，但只导出当前已使用且强制同时输出 Q/R 的 `aclnnLinalgQr`；没有同语义 Q-only、Geqrf/Orgqr 或更快 primitive。自写 QR 会改变持久 raw Q、排序后 optimizer state 与 checkpoint/resume 数值轨迹，裁决 `NO_GO_NO_EQUIVALENT_Q_ONLY_OR_FASTER_PRIMITIVE_IN_FIXED_ENV`。
- 稳定Step亲和API复核：`addmm`已经是单个ACLNN融合kernel且纯kernel仅15.089ms；`npu_confusion_transpose`相关不同copy族纯kernel宽松合计21.941ms、固定A3又无ACLNN实现；`npu_add_layer_norm`把全部Add与LN-forward错误全算可消除的极端上限也仅21.007ms。三者均低于22.7ms或缺少唯一兼容边界，不进入机制/训练门禁。

## 9. 下一步开发顺序

1. **冻结交接状态。** 以 `f922c38` 和 STEP-189 脱敏 Top N 表为唯一当前起点；运行前重新核验权威仓库 clean、正确容器、后 8 卡、8 rank、端口和 `npu-smi` 进程。
2. **已关闭P0：冻结图像 Backbone/FPN Conv-BN。** 43对在8-rank真实shape上可节省51.058ms（1.1858x），但最大NRMSE=`1.991e-3`超过严格`1e-4`门槛，裁决`REJECT_NUMERIC_GATE_NO_TRAIN_NO_COMMIT`。
3. **已关闭P1：`BatchMatMulV2`。** 82.9ms定位到已正式拒绝的point_sampling边界；broadcast显著回归，packed-BMM虽函数级exact但两轮30-step端到端回归，不重开。
4. **P2：普通步underfeed现有证据复核。** 仅使用已有脱敏全阶段栈和STEP-189摘要定位host launch gap、同步点、DDP与Python调度；因原始profile已删除，不以猜测补造调用边界，也不重复采集。
5. **Index/Reduction复核。** 只检查是否存在尚未被MapTR、grad-clip、SOAP等历史证据覆盖的新边界；没有独占收益超过22.7ms的对象即关闭。
6. **SOAP 周期线。** 只有找到当前固定版本可用、逐状态等价且 checkpoint 可恢复的实现才重开；禁止为追求性能升级远端组件或改变预条件器数学语义。
7. **每个候选的验收。** 先 8 卡 30-step profiler-off；保留项再按风险决定是否做 876-step、resume 和同 checkpoint 推理。结果同时报告相对直接父提交的增量和相对 `63861df/4c37039` 的累计收益。
8. **独立修复最终评测入口。** `lidar_type` 配置兼容和评测数据可达性属于正确性基础设施，不与性能算子 commit 混合；修复后再对保留的 checkpoint 执行正式任务指标评测。

### STEP-197选择性Conv-BN补充

为排除“全量43对融合失败但部分stage可用”，已在同一正确容器、后8逻辑NPU、8 rank、同checkpoint和真实`[112,3,576,1024]`输入下，分别测试`stem/layer1/layer2/layer3/layer4/FPN`六组。六组shape/stride/finite及原state/optimizer引用均通过，但NRMSE为`2.977e-4～1.529e-3`，净节省仅`5.865～12.185ms`；没有单组同时满足`NRMSE<=1e-4`和`>22.7ms`，因此未测试组合，裁决`REJECT_NO_SELECTIVE_GROUP_MEETS_NUMERIC_AND_22P7MS_GATE`。

截至STEP-197，现有一次稳定全阶段profile可导出的独立TopN机制均已采用、正式拒绝或因固定环境/状态等价边界关闭。当前无新业务commit，HEAD保持`f922c38`；无profiler普通步NPU/GPU吞吐比约0.700、完整稳定周期约0.510，1:1目标尚未达成。

### STEP-198/199 图执行与 SOAP QR 自定义算子补充

- STEP-198 原生 NPUGraph 对冻结图像塔保持逐元素一致，但完整 `copy_+replay` 比 eager 慢约`0.714ms`并额外预留约`30.667GB`，已拒绝。
- STEP-199 固化了 SOAP QR 的严格状态合同：raw `Q`、stable sort、连续至少两个 QR 周期的完整 optimizer state/参数及中途 resume 必须逐位一致；仅正交、重构或 sign-aligned 容差不足以准入。固定环境未发现满足该合同的更快底层 primitive，因此不编译、不占 NPU、不改业务、不提交。
- 截至 STEP-199，当前一次稳定全阶段 profile 可唯一归因且符合固定环境规则的候选已全部采用、正式拒绝或 NO_GO；1:1 目标仍未达成，不能以放宽 loss/optimizer/checkpoint 数值门禁制造收益。
- STEP-200补齐早期advisor的三类亲和API缺口后仍无新GO项；advisor规则库版本旧于客户固定环境，其次数与host聚合不能覆盖当前稳定Step的纯kernel和端到端反证。

### 当前阻塞状态

1:1目标没有完成。唯一稳定全阶段profile的可唯一归因候选已全部采用、正式拒绝或因固定环境/严格状态等价NO_GO；STEP-198图捕获、STEP-199自定义QR与STEP-200亲和API连续复现同一阻塞。继续推进至少需要以下一项外部条件变化：

- 厂商在当前兼容栈提供与现有算子同语义、可通过raw state门禁的更快primitive；
- 客户明确授权改变当前禁止变更的软件栈，并提供对应兼容/回退/验收环境；
- 客户授权在已删除的16.65GB原始timeline不可恢复的前提下重新采集一次新的稳定全阶段profile，用于把约5.85s分散underfeed唯一归因到新源码边界。

在上述条件出现前，不允许通过改变loss、SOAP数学、optimizer/checkpoint状态、batch/rank或GPU对齐合同制造1:1结果。

## 10. 执行约束速查

- 连接信息只从本地 `机器IP.md` 读取，不复制到文档、日志或 commit。
- 不从远端下载日志、profile、数据集、权重或 checkpoint；本地只保存脱敏统计。
- 启动命令继续复用远端已验证的 canonical `ddp_train_30.sh`，其已记录 SHA256 为 `10ad92c...e0fc`；不要自行重拼训练入口。
- 正式配置、batch、worker、prefetch、SOAP 频率和随机性字段必须在每次 A/B 前重新断言。
- 测试结束恢复 `fusion_result.json` 等运行副作用，删除 `kernel_meta`、临时脚本、原始日志和不再需要的 profiling 数据，并复核进程、端口和 Git 状态。

## 11. 2026-08-14 GPU合同对齐后的权威覆盖项

- 用户指定的本地GPU配置与GPU `loss.log` 运行合同已重新对齐：seed0、deterministic=False、batch/rank=16；此前“配置尚未严格对齐”和0.553旧条件比不再作为当前结论。
- 当前权威profiler-off结果：稳定普通步NPU/GPU吞吐比约0.700；包含SOAP的完整稳定周期约0.510。目标1:1仍未达到。
- 稳定step后仅采集一次rank0带栈/shape连续窗口；同一trace覆盖SOAP和普通步及全部训练阶段。最终TopN、异常报告、10节架构报告、v4栈归因和候选矩阵均已完成。
- STEP-189之后计划中的“若证据不足再采最小host stack”已由本次唯一带栈profile覆盖，不再追加第二次采集。Conv/TransData、BMM、Index族按新证据复核后没有新的安全独立候选；不得按旧下一步列表直接启动实验。
- 本条的`16,647,868,129 bytes/raw=0`只描述用户要求长期保留之前删除的旧profile。STEP-202获准重新采集后，新raw已永久原位保留：205个文件、`16,647,970,748 bytes`，retention manifest SHA=`464af966...d350`且`retained=true/deletion_authorized=false`；业务HEAD `f922c38` clean。下一动作仍必须以新候选的状态/功能等价证据为前提，不能为了性能改变loss或optimizer语义。
- STEP-194纠正BMM归因：普通步约82.9ms的单次`aclnnMatmul`是`BEVFormer.point_sampling`，不是loss/target的280次小BMM。broadcast方案83.394→207.551ms回归；packed-BMM虽函数exact且83.410→1.856ms，但两轮正式30-step端到端回归，保持关闭。ViewCopy可定位的最大单次clone仅6.888ms，聚合项无唯一安全边界，亦关闭。
- STEP-198原生NPUGraph已在同checkpoint、8-rank、真实图像shape上完成机制门禁：输出/状态完全一致，但eager `326.609ms`、graph含输入copy+replay `327.281ms`，净收益`-0.714ms`、`0.99795x`，capture额外reserved约30.667GB。裁决`REJECT_MECHANISM_GATE_NO_TRAIN_NO_COMMIT`；不短训、不改业务、不commit。live rank/端口证据具备，但`npu-smi`未返回匹配PID，明确记为证据缺失。raw/harness已清理，仅留脱敏summary/manifest，远端clean/进程0/端口0。

### STEP-203 GPU/NPU trace 同义业务对比

- GPU无栈trace在远端安全解包并永久保留；稳定普通组15步中位service/device-wall/busy/kernel=`5848.556/5846.212/2045.559/2045.559ms`。NPU带栈普通三步kernel中位1817.157ms并不高于GPU总kernel，profiler-off NPU/GPU普通步仍为1.429，说明差距集中在host/underfeed和NPU特有高频路径，而非整体device compute。
- 新P0为`bev_encoder.random_spatial_mask`：GPU相同配置下`fill [1,8,8]`及CUDA Fill kernel每稳定普通步严格2048次、8.194ms；NPU同业务AICPU ViewCopy每步2048次、96.610ms，11.79x且直接超额88.416ms/步。GPU无stack，源码归因标注为高置信推断而非行级直接证明。
- 候选仅进入严格机制门禁：必须保持随机抽样、RNG state、重复block覆盖、mask/output逐位值、dtype/layout/alias/autograd，再经过测试集、loss/grad、正式8-NPU A/B和长训/checkpoint门禁；当前未改码、未训练、未commit。完整对比见`STEP-203_GPU_NPU_profiling对比报告.md`。
- MSDA残余差距不重复已采用实现；SOAP QR、Conv-BN、channels-last、point_sampling BMM、Index/Reduction聚合、NPUGraph及三类亲和API继续按历史矩阵关闭。P0完全回收96.610ms后普通步估计仍约为GPU的1.407x，1:1目标尚未完成。

### STEP-204 random_spatial_mask 当前状态

- 低分辨率`index_fill_+repeat`候选已通过CPU64/64、后8 NPU 8×64及真实业务函数逐位/RNG/alias门禁；业务patch仅1文件且未提交。完整同步边界净省206.246～235.591ms，保守纯kernel下界95.669ms。
- fresh 30-step baseline/candidate均exit0；稳定普通mean `5.322551→5.000786s`、吞吐`+6.434%`，14/14配对步加速。loss最大相对偏差0.3934%、finite grad最大2.0626%，dynamic loss-scale相位一致；candidate普通NPU/GPU吞吐比约0.8647，仍未达到1:1。
- paired resume均从各自iter30恢复到iter36并通过：loss全finite，Iter31共同grad inf/scale下降/skip；meta29→35、559个optimizer step26→32、checkpoint schema/shape/dtype/finite一致。resume loss最大相对差0.5037%；finite grad最大绝对相对差11.2622%发生在Iter34，实际48.30773→42.86721。
- 固定512测试尚未执行。STEP-185原512源身份工件已按旧生命周期删除，当前只剩basename/bytes/dataset_len，没有源内容SHA或first512首尾ID；明确路径和容器mount均无法唯一恢复同一镜像。严格门禁要求不猜数据、不换测试集，因此当前停止在数据身份恢复，未启动512、876或commit。

#### STEP-204 最终裁决（取代上述阶段性“当前状态”）

- fresh paired 876-step已完成并反证30-step短窗收益：candidate相对baseline的stable normal慢`1.071883%`、完整cycle慢`0.081110%`、all1～875慢`0.096447%`、末100普通慢`2.277711%`；SOAP虽快`1.583035%`，不足以满足普通步/完整周期采用门禁。candidate all1～875吞吐`16.525519 samples/s`，相对既有GPU前876吞吐`28.346540`仅`0.582982:1`。
- 最终裁决：`REJECT_LONG_RUN_NO_SUSTAINED_NORMAL_OR_CYCLE_GAIN_NO_COMMIT`。30-step稳定普通吞吐`+6.434%`归类为未被长期训练复现的短窗假阳性；不再启动876 resume或固定512，不创建commit。
- 最终单文件diff已原位永久保存，SHA256=`921d53daa0af10386843acbe1fabd712567e22a4cf8208e3204a8f33aed30313`。权威业务文件已精确恢复HEAD blob=`5423a7d7...cde`、文件SHA=`399f349d...6ae2`；权威仓库和baseline worktree均tracked clean，端口/训练/NPU进程0。
- baseline/candidate的30-step、paired resume、876-step日志/metrics/checkpoint、比较工件、所有失败启动日志、最终rejected patch，以及用户指定长期保留的NPU/GPU profiling原始数据均未删除、移动或拉取本地。

### STEP-205 MSDA 残余差距裁决
- GPU/NPU 6 次调用已按 shape/count/dtype 对齐；空间 FP32 位置贡献约 `+81.214ms/step`，但两端语义调用均为单一主 kernel，并非 launch 碎片。
- 项目可控的 Cast/Copy/Zero 等纯 device 乐观上限约 `6.236ms/step`，低于 `22.7ms`；无 TransData/AICPU/同步可消除。
- DrivingSDK runtime 未暴露 im2col_step/tiling/workspace/layout/precision 参数，包内无同语义替代 primitive；主算法属于固定 SDK 黑盒。
- 最终裁决：`NO_GO_MAIN_KERNEL_FIXED_SDK_NO_PROJECT_CONTROLLED_EQUIVALENT_BOUNDARY`。无训练、无 NPU 执行、无重采 profile、无业务/环境改动、无 commit；永久 raw 保持原位。

### STEP-206～208 当前最终状态

- 剩余TopN和阶段差距均已按四时钟、wait-anchor、源码consumer与GPU同义族复核；没有新的项目可控、单一、严格等价、可独立回退且理论净收益`>22.7ms/step`边界。
- `PER_STREAM_QUEUE=1`虽存在于固定2.7.1，但当前无多线程、多非HCCL compute stream的Dequeue独占阻塞越线证据，不作为A/B候选。
- 最长1～875 step口径下，当前HEAD NPU/GPU吞吐=`16.541460/28.346540≈0.5835:1`，平均step仍差`3.222591s`；1:1目标未完成。
- STEP-209第三次只读复核确认外部状态、固定API能力与永久数据均未变化；STEP-207～209同一阻塞连续三次成立。目标未完成，现正式状态为`blocked`，裁决`BLOCKED_THIRD_CONSECUTIVE_AUDIT_NO_EXTERNAL_STATE_CHANGE`。
- 精确重开输入仅为：schema兼容且同语义更快的MSDA空间FP32 primitive、逐位Q-only QR primitive、非HCCL多stream Dequeue越线直接证据，或包含具体版本/API/兼容/隔离验收/回退方案的软件栈授权。
