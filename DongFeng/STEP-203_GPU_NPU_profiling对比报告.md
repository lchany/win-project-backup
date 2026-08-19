# STEP-203 GPU/NPU profiling 对比报告

日期：2026-08-15  
范围：只读复用客户 GPU 无堆栈 trace 与本轮 NPU 稳定全阶段带栈 profile；未重新采集、未启动训练、未修改业务代码。

## 1. 对齐合同与限制

- GPU 与 NPU 使用同一客户配置文件、8 卡、batch/rank=16、seed=0、deterministic=False、SOAP frequency=10；`lidar_dropout_prob/lidar_spatial_rate/lidar_mask_ratio=0.1/0.2/0.2`。
- GPU trace 共有 50 个完整 `ProfilerStep` 窗口，但标签为两次 Step0 加 Step1～48，不能按文件名 `iter0_49` 假定有 Step0～49。稳定普通组取 Steps33～41、43～48（15 步），SOAP 组取 Steps22/32/42（3 步）。
- NPU 普通组取本轮连续稳定 Steps24～26（3 步），SOAP 取 Step23。GPU/NPU step 编号不机械相等，而按 SOAP 相位、调用次数、shape、配置和业务边界对齐。
- GPU 没有 Python stack，也没有 Ascend `Wait Time/Total Cost` 语义；源码归因最高只能是“shape+次数+配置+源码版本+阶段”的分层推断。NPU 带栈 service/wall 受 profiler 严重扰动，吞吐结论仍以 profiler-off 基线为准。
- 统计口径：GPU 普通组为 15 步中位数；NPU 普通算子数据为 3 步均值或中位数（表中注明）。kernel sum 允许重叠重复累计，busy union 去重叠；两者均不能直接当端到端可回收 wall。

## 2. GPU 基线与四时钟

| 场景 | service | device wall | busy union | kernel sum | underfeed | GPU wait |
|---|---:|---:|---:|---:|---:|---|
| GPU 稳定普通 15 步中位 | 5848.556 ms | 5846.212 ms | 2045.559 ms | 2045.559 ms | 3820.300 ms / 65.367% | N/A |
| GPU SOAP 3 步中位 | 8267.400 ms | 约 8267 ms | 3997.019 ms | 3997.019 ms | 4145.577 ms / 51.455% | N/A |
| NPU 稳定普通 3 步中位 | 48390.475 ms | 46576.717 ms | 1794.662 ms | 1817.157 ms | 约 96.23% | 有，但不可与 GPU N/A 对比 |

NPU/GPU 的 profile 内普通步 ratio 分别为 service 8.274、wall 7.967、busy union 0.877、kernel sum 0.888。service/wall 的巨大倍数主要是 NPU `with_stack+record_shapes` 扰动，不代表真实训练比；相反，profiler-off 权威普通步为 NPU 6.1796 s、GPU 4.3241 s，NPU/GPU=1.429，吞吐比约 0.700。

## 3. 阶段对比

| 阶段 | GPU 普通15步中位 | NPU 本轮4步栈均值估计 | NPU/GPU | 置信度/解释 |
|---|---:|---:|---:|---|
| forward+loss | 1034.268 ms | 1302.851 ms | 1.260 | 中；NPU为SOAP+3普通步的调用栈均值，GPU为普通步时间边界 |
| backward | 656.097 ms | 402.307 ms | 0.613 | 中；NPU为SOAP+3普通步均值，仍可说明差距不来自整体backward device compute |
| optimizer（普通） | 319.422 ms | N/A | N/A | NPU 4 步栈聚合混入 SOAP QR，不伪造普通值或ratio |
| SOAP optimizer/QR | QR 1198.255 ms | QR 22798.071 ms | 19.03 | 高；同为周期 QR 族，但固定环境严格状态等价路线已关闭 |

NPU 普通 kernel sum 反而低于 GPU，而 profiler-off service 慢 42.9%，证明主要方向应是 NPU 特有的高频 AICPU/host 同步与少数同义算子超额，不能把所有 device kernel 总量机械优化。

## 4. 同义算子/业务族 TopN 差距

| 同义业务族 | GPU 普通 | NPU 普通 | NPU/GPU | NPU 可解释超额 | 结论 |
|---|---:|---:|---:|---:|---|
| `random_spatial_mask` 8×8 slice fill | CUDA Fill kernel 8.194 ms，2048 次/步 | AICPU ViewCopy 96.610 ms，2048 次/步 | 11.79 | 88.416 ms/步纯 kernel | 新的唯一安全边界，P0 |
| MSDA forward | 30.452 ms，6 次/步 | 约 81.81 ms，6 次/步 | 2.687 | 约 51.36 ms/步 | 已采用 DrivingSDK 融合实现；只允许新的严格等价底层机制 |
| MSDA backward | 146.352 ms，6 次/步 | 约 186.86 ms，6 次/步 | 1.277 | 约 40.51 ms/步 | 同上，不能重复已有迁移 |
| SOAP QR | 1198.255 ms | 22798.071 ms | 19.03 | 约 21600 ms/SOAP 步 | 最大周期差距，但固定环境无逐状态等价 primitive，保持关闭 |
| Conv/BN/布局整体 | Conv 356.708 + norm 9.451 ms | Conv 主体、TransData、BN 等分散，未高于 GPU 同族总量 | 不构成 NPU 超额 | 0（不宣称可回收） | channels-last 不支持；Conv-BN 数值门禁失败 |
| MatMul/BMM | GPU 全族 520.407 ms | NPU 单次 point_sampling MatMul 约 82.75 ms | 不可机械比 | N/A | packed-BMM 函数级快但两轮端到端回归，关闭 |
| Index/Reduction | GPU index 95.841 + reduction 73.342 ms | NPU 多调用点、wait 占比高 | 不可机械比 | N/A | 无新的唯一功能边界，历史关闭 |

`random_spatial_mask` 的 GPU 证据链为：15 个稳定普通步中，`aten::fill_` 输入 `[1,8,8]` 每步严格 2048 次，host 中位 22.645 ms；同阶段 `select [16,1,128,320]`、`slice [1,8,320]`、标量 `floor_divide/remainder` 也分别严格 2048 次；CUDA `FillFunctor<float>` kernel 每步严格 2048 次、8.194 ms。结合相同配置、相同函数版本与 NPU line427 四步共 8192 次闭合，业务归因置信度为高；由于 GPU 无堆栈，仍不表述为 GPU 源码行级直接证明。

## 5. 优先级、优化理由与门禁

### P0：向量化 `random_spatial_mask` 的 2048 次小 slice 写

- 原因：这是本轮第一个同时满足单一源码边界、固定 shape/count、NPU 显著超额且 GPU 同业务基线可对齐的新候选；NPU 纯 kernel 理论上限 96.610 ms/普通步，若只追平 GPU 对应 kernel，直接上限约 88.416 ms/普通步。host/sync 可能还有收益，但未形成非重叠 wall 证据前不得叠加到上限。
- 如何优化：优先构造批量 block 坐标和一次性布尔/索引 mask，再以一个或少量 AI Core 友好操作完成写入；禁止改变被选择 block 的集合、循环顺序导致的覆盖语义或 RNG 消耗序列。候选必须单独 commit、独立回退。
- 功能风险：最高风险是随机数消费数量/顺序变化、重复 block 覆盖、边界索引、dtype/layout、in-place alias 与后续 loss 轨迹变化。
- 门禁：先用固定输入和固定 RNG state 做原函数/候选逐元素 exact；断言调用前后 RNG state exact、shape/dtype/stride/device/alias 合同、重复 block 与边界样例；再用测试集及相同 seed 检查输出/loss/grad finite 和严格容差；机制收益须超过既定 22.7 ms 噪声门槛后，才允许 8-NPU 30-step A/B，随后长训、checkpoint/resume 与推理回归。不得用平均 loss 接近替代逐阶段正确性。

### P1：MSDA forward/backward 残余差距

DrivingSDK 实现已经采用且通过完整门禁。只有发现当前实现内部新的单一融合/launch 边界，并能保持 forward、backward、梯度、AMP 与 checkpoint 语义时才重开；不得把“再次切换 DrivingSDK”作为新方案。

### 关闭/不重试矩阵

| 方向 | 状态 | 原因 |
|---|---|---|
| SOAP QR | `CLOSED_NO_NEW_FIXED_ENV_EQUIVALENT` | raw Q、排序、optimizer state、连续周期及 resume 必须等价；固定环境无更快同语义 primitive |
| Conv-BN fold | `REJECT_NUMERIC_GATE` | 真实 8-rank 最大 NRMSE 1.991e-3，高于 1e-4 |
| channels-last/TransData | `NO_GO_UNSUPPORTED` | 固定 torch_npu 版本不支持目标 memory format |
| point_sampling packed-BMM | `REJECT_END_TO_END` | 函数 exact 且微基准快，但两轮正式 30-step 均回归 |
| Index/Reduction/ViewCopy 旧聚合 | `CLOSED_NO_UNIQUE_BOUNDARY` | 跨功能调用点且 wait 不能当纯计算收益；本轮 mask 2048 次是新独立边界，不与旧聚合混算 |
| NPUGraph | `REJECT_MECHANISM_GATE` | exact 但含 input copy+replay 慢约 0.714 ms，额外预留约 30.667 GB |
| addmm/confusion-transpose/add-layernorm | `NO_GO_BELOW_THRESHOLD_OR_UNSUPPORTED` | 已融合或理论上限低于 22.7 ms，且无唯一兼容边界 |

## 6. 可回收上限与对 1:1 目标的意义

- 新 P0 直接纯 kernel 上限：96.610 ms/普通步；按 GPU 同义 kernel 对齐的现实差额约 88.416 ms/普通步。
- MSDA 可解释 device 差额约 91.87 ms/普通步，但现行优化已采用，不能把它当成尚未实现的可回收承诺。
- 即使完全回收 P0 的 96.610 ms，profiler-off 普通步 6.1796 s 约降至 6.0830 s，距 GPU 4.3241 s 仍约 1.407×；因此 P0 值得优先做，但单独不足以达成 1:1。
- SOAP QR 理论差额最大，但因功能/状态等价门禁关闭，不能计入可执行收益计划。

结论：下一项唯一合理的实施候选是 `random_spatial_mask` 批量化；先做不占正式 8 卡的严格等价机制门禁，再决定是否进入客户配置下的 8-NPU A/B。其余方向按历史矩阵交叉检查，不重复用微基准覆盖端到端反证，也不为性能改变 loss、随机性、SOAP 或 checkpoint 功能。
