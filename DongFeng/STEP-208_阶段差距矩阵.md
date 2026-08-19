# STEP-208 NPU/GPU 阶段差距矩阵

裁决：`NO_GO_NO_NEW_PROJECT_CONTROLLED_SINGLE_EQUIVALENT_BOUNDARY_ABOVE_22P7MS`。

本轮只读复用 STEP-202 NPU 稳定普通 Step24～26、SOAP Step23及STEP-203 GPU稳定普通15步、SOAP 3步；未训练、未调用NPU、未重采、未改删永久产物。

## 权威四时钟

| 组 | service | device wall | busy union | kernel sum | total-cost / wait |
|---|---:|---:|---:|---:|---:|
| NPU普通3步均值 | `47606.668ms` | `46449.972ms` | `1795.134ms` | `1814.429ms` | `48268.775ms`，含大量wait |
| GPU普通15步中位 | `5848.556ms` | `5846.212ms` | `2045.559ms` | `2045.559ms` | GPU wait=N/A |
| NPU SOAP Step23 | `71150.834ms` | `68064.002ms` | `24615.823ms` | `24628.128ms` | `68144.011ms` |
| GPU SOAP 3步中位 | `8267.400ms` | `8266.343ms` | `3997.019ms` | `3997.019ms` | GPU wait=N/A |

带栈NPU service/wall受严重扰动，只用于归因；profiler-off普通NPU/GPU=`6.1796/4.3241=1.429`，完整稳定周期均值=`8.6575/4.416s`，吞吐目标仍未达到1:1。

## 阶段矩阵

| 阶段 | NPU证据 | GPU基线 | 差距解释 | 单一边界与历史状态 | 裁决 |
|---|---|---|---|---|---|
| data / prelaunch | 普通prelaunch均值`1630.534ms`，Step24～26=`1619.191/1709.697/1562.714ms`；scatter每步457次，device-self约`0.008ms/step` | 同样457次record_stream/step；最大图像copy同shape同为1次/step | NPU host时钟被with_stack放大，不能当净收益 | DataLoader/pin/affinity及DataContainer pin均已正式关闭；pin正式A/B使full step `+3.475%`、吞吐`-3.358%` | 无新边界 |
| forward + loss | `1302.851ms`四步栈均值估计 | `1034.268ms`普通步边界 | NPU约`+268.583ms`，但跨大量consumer | random mask长期反证；MSDA固定SDK NO_GO；point_sampling、Conv/BN/layout、Index/Reduction/Unique、ViewCopy等已关闭。其余项目Matmul/BMM单点最大`10.753ms` | 无新边界 |
| backward | `402.307ms` | `656.097ms` | NPU整体反而约快`253.790ms` | 最大NPU纯kernel仍是MSDA grad `186.862ms/step`，但STEP-205证明项目可控内部仅`6.236ms/step`；其余Conv backward等必要数学或历史关闭 | 非差距来源 |
| optimizer普通 | NPU四步栈聚合混入SOAP QR，不能伪造普通phase值 | `319.422ms` | 现有证据不足以构造严格ratio | grad-norm/scalar sync/zero_grad/DDP/foreach/Graph等均已有采用或正式拒绝；wait/bubble不能聚合 | 无新证据，保持关闭 |
| SOAP optimizer | QR `22798.071ms/SOAP step` | QR `1198.255ms/SOAP step` | 约`+21.600s`，最大周期差距 | 固定环境只有强制同时输出Q/R的`aclnnLinalgQr`；block/geqrf/orgqr/out-buffer/multistream均已拒绝，替换会改变raw Q、排序、optimizer state或环境 | `NO_GO_FIXED_ENV_STATE_EQUIVALENCE` |
| communication | 四步通信kernel合计`33.058ms`，约`8.264ms/step`；compute overlap `80.509%`，未掩盖上限约`1.611ms/step` | 稳定步`nccl:all_reduce`同为13次/step；GPU无Ascend wait语义 | 量级远低门槛，且大部分掩盖 | DDP/HCCL路径已审计；不能用total-cost wait制造收益 | 低于门槛 |
| tail / logger | 普通tail均值`65.502ms`；仅Step24保留`194.568ms`大tail，logger栈约104ms、TensorBoard add_scalar仅6.265ms；Step25/26未越bubble保留阈值 | 无同口径显式tail | tail是最后kernel后的host/wait边界，不是纯device算子 | TextLogger显存同步降频已采用；剩余日志、untraced blocking不满足纯device收益和功能不变门禁 | 无新边界 |

## 证据缺口

- GPU无Python stack和Ascend wait/total-cost，不能伪造GPU源码行或直接比较wait。
- NPU `with_stack + record_shapes`严重放大host/service/wall，不能从prelaunch、host-self或tail直接推导profiler-off收益。
- NPU普通optimizer phase没有不混入SOAP QR的权威阶段时钟；不足以授权猜测性修改。
- 阶段总差、bubble及无项目栈条目均跨consumer，不能合并成一个候选。

只有出现新的项目行级单consumer证据、固定SDK同语义更快primitive，或客户授权的软件栈能力时才可重开。
