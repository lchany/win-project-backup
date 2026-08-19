# STEP-206 剩余 TopN 与 GPU 基线关闭矩阵

## 结论

裁决：`CLOSED_NO_NEW_SINGLE_EQUIVALENT_BOUNDARY_ABOVE_22P7MS_AFTER_GPU_BASELINE_CROSSCHECK`。

本轮只读复用 STEP-202 永久保留的稳定全阶段 NPU profiling 和 STEP-203 永久保留的 GPU 无栈 trace；没有训练、调用 NPU、重新 profiling、修改业务/环境、删除、移动、覆盖或拉取远端产物。带栈普通步的 service/wall 只作归因，最终性能仍以 profiler-off NPU/GPU `6.1796/4.3241=1.429`、吞吐比约 `0.700` 为准。

## 四时钟与准入门槛

稳定普通 Step24/25/26 的 `service / device wall / busy union / kernel sum / total-cost`：

- Step24：`48390.475 / 46576.717 / 1794.662 / 1807.173 / 48394.209 ms`；
- Step25：`48482.702 / 46826.372 / 1803.371 / 1817.157 / 48723.059 ms`；
- Step26：`45946.827 / 45946.827 / 1787.368 / 1818.956 / 47689.057 ms`。

准入只接受单一、严格等价、可独立回退且纯 device 理论净收益 `>22.7 ms/step` 的源码边界；跨消费者聚合、host profiler 开销、排队 wait 和整段 bubble 均不算候选收益。

## 剩余 TopN 关闭矩阵

| 对象 | 每步count | 纯kernel/step | wait/step | shape / stack / consumer | GPU对齐与历史状态 | 裁决 |
|---|---:|---:|---:|---|---|---|
| MSDA backward | 6 | `186.862ms` | 可忽略 | 固定DrivingSDK backward，最大为spatial FP32调用 | STEP-205证明残差在SDK单kernel | 项目可控内部仅`6.236ms`，NO_GO |
| Conv | 多层 | `133.208ms` | 非主因 | 冻结backbone/FPN多消费者 | channels-last/HF32/Conv-BN已拒绝 | 无新等价实现 |
| AICPU ViewCopy | 2048 | `96.610ms` | `11.822ms` | 全部闭合到`random_spatial_mask` | STEP-204 fresh 876-step反证并回退 | 旧拒绝边界 |
| point_sampling Matmul | 1 | `82.985ms` | 非主因 | `bevformer_encoder.py:240` | packed-BMM已正式E2E反证 | 唯一越线Matmul，但不是新边界 |
| MSDA forward | 6 | `81.810ms` | `4.999ms` | 固定DrivingSDK forward | STEP-205 NO_GO | 项目边界低于门槛 |
| Conv TransData | 多层 | `67.011ms` | `63.728ms` | Conv内部格式、多消费者 | internal-format/channels已关闭 | 非单一源码边界 |
| Index/ReduceSum/Unique | 5205/2313/512等 | `55.007/54.740/48.243ms` | `3418.736/493.934/624.211ms` | loss/target与forward多消费者 | 历史索引/归约/Unique关闭 | wait主导，不可聚合 |
| AI-core ViewCopy | 多消费者 | 聚合`31.429ms` | 聚合`534.062ms` | random mask外最大无项目栈上界`20.833ms`；PillarVFE/map loss/geo loss为`6.614/3.755/3.216ms` | random mask已拒绝 | 无新单点越线 |
| addmm/LN/transpose | 多消费者 | 既有严格上限均`<22.7ms` | 不计wait | BEV/decoder分散调用 | STEP-200关闭 | 不重开 |

旧`viewcopy_stack_distribution_v2.json`把Host/Device self展示字段互换；原始`operator_details.csv`复核显示`aclnnInplaceCopy`的`Device Self Duration With AICore`为0。本轮不覆盖永久保留JSON，只记录纠错，并以每个窄项目栈全部InplaceCopy device-self作为保守上界。

## prelaunch / scatter / H2D

普通Step24/25/26 prelaunch为`1619.191/1709.697/1562.714ms`，首kernel均为Cast，host栈落到`train→train_step→scatter→scatter_gather.py`。每步严格457个copy/record_stream；NPU带栈scatter host-self合计`1098.397ms/step`，device-self仅约`0.008ms/step`，host数字受堆栈采集显著放大。

- `[16,1,7,3,576,1024]`：1次/step，NPU带栈copy host-self `105.499ms/step`；GPU同shape同样1次/step，GPU host中位约`124.095ms`；
- `[573440,7]`：32次/step、`96.197ms/step`；`[245760,7]`：32次/step、`45.125ms/step`；`[275,120,9]`：48次/step、`23.999ms/step`。

GPU稳定15步的record_stream也是严格457次/step，说明双方业务结构一致。DataContainer pin既有30步使full step变慢`3.475%`、吞吐下降`3.358%`，DataLoader/pin/affinity也已关闭。现有数据不能给出profiler-off单copy H2D净收益，因此无GO。

## forward + loss 拆分

GPU forward+loss约`1034.268ms`，NPU带栈阶段估计约`1302.851ms`，差约`268.583ms`，但跨大量consumer。最内层源码拆分后：

- `bevformer_encoder.py:240 point_sampling`：1次/step、纯device `82.985ms`、host-self `0.127ms`，为已拒绝packed-BMM；
- `maptr_decoder.py:752` Matmul `5.172ms/step`，`temporal_self_attention.py:286`为`0.428ms/step`；其他无项目栈Matmul合计`6.253ms/step`、跨22次；
- `geo_loss.py:58` BatchMatMul `10.753ms/step`；无项目栈BatchMatMul聚合`9.980ms/step`；
- 无项目栈Mm聚合`27.908ms/step`但跨207次/多个consumer，不能当一个可回退边界；其余主要Mm属于SOAP历史关闭路径。

其余device Top项均为Conv、InplaceCopy、MSDA、Nonzero、Index、ReduceSum、Unique、BN、Mul、IndexPut等已关闭家族。select/slice/index等巨大host-self条目device-self为0且受with_stack放大，不能据此推导profiler-off净收益。

## wait、bubble与证据缺口

Fill、Broadcast、Index、MemSet、Equal、Cast、Nonzero MemSet、InplaceZero等total-cost Top有`94%～99.9%`等待；不能把等待汇总当可删除kernel。最大约`53～56ms`的VectorNorm→Stack和Add→NeTensor bubble分别交叉到既有grad-norm/scalar sync/DDP路径；bubble本身不是单一consumer。

GPU无栈，只能用shape/count/阶段映射，不能伪造源码行；NPU带栈严重扰动host与service/wall；无项目栈聚合仍可能跨consumer。只有出现固定SDK同语义更快primitive、客户授权的软件栈能力，或新的低扰动单边界证据时才可重开。
