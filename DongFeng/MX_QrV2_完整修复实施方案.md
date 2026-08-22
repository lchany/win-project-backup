# MX QrV2 完整修复实施方案

状态：v3/v4 均已被真实设备 finite 门禁证伪；v5 首次真实调用发生 AICore timeout/trap。STEP377 attempt6～8 均在 QrV2 数值执行前因实验链合同错误退出，不构成算子根因证据；attempt9 的8/8 rank已越过ready/ownership/gate，profile中观察到了尚未闭合rank覆盖和具体AIC身份的generic `QrV2` marker，但0/8产生done，无summary、Q/R或finite记录。它只证明delta1-only未形成正常完成证据，不能证明8/8进入concrete kernel、QrV2返回、终止阶段、底层指令根因或已修复。当前controller已重新锁定，attempt9永久退役。
日期：2026-08-22
适用包：`cann-driving-cloud-ops-26.0.7-a3-cann8.3.rc1-torch2.7.1-py3.11-aarch64.zip`

> **当前执行指针：只有第 17 节是当前可执行方案。第 4～16 节中所有 v1～v5 identity、SHA、补丁集、构建、发布和“唯一路径”均是历史实验记录，不得用于新的构建、NPU 测试、训练或发布。**

## 1. 结论与目标

本方案记录的 v5 是已完成真机证伪的历史候选，不再是可发布候选。它在已经真机证伪的 v4 之上只修正两项 Matmul 存储位置合同：`CalcQForLARFB()` 第二次 `vtvMatmulObj` 直接使用 `vLocal(VECIN)`；`qaMatmulObj` 的 A/B/C 声明改为 `VECIN/GM/VECIN`，与 LARFB/SSRFB 现有实参一致。v5 保留 v4 的其余生命周期、zero-work、const 兼容、`alphaBuf`、`UpdateAForLARFB()` 事件和外层 `SyncAll()` 修改，不替换 QR 算法，不改变 FP32、64×64 分块、shape/dtype、workspace 总量、输出初始化或 Python API。wrapper 的既有分支条件是 `min(m,n) <= 80` 走 torch、`min(m,n) > 80` 走 MX；正式版本仍要求所有满足 `min(m,n) > 80` 的调用执行最终通过全部门禁的 MX QrV2，禁止额外 fallback 掩盖问题。

发布必须同时满足五类门禁：源码范围、CPU 官方算子数学对照、设备端具体 AIC 身份、30-step 逐步 loss、性能。任一门禁失败都不能宣称修复完成。

## 2. 已确认事实与证据边界

### 2.1 已确认事实

- ZIP 内目标源码相对路径为：
  `mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.cpp`
- 已审计原始源码 SHA256 为：
  `2dbaf1e1b5383563c23cdac7a5151b14605f8585b6e48fd8c58065fb5c1206c9`
- 原始 `Process()` 在 `LARFB()` 返回后调用 `CalcQForLARFB(false)`。
- 原始 `LARFB()` 的 active 和 inactive 分支都已释放 `tLocal/vLocal/aLocal`。
- `CalcQForLARFB(false)` 会再次读取 `tLocal/vLocal`。
- 对 192×192 输入，`blockSize=64`、`blockp=3`。最后 panel 为 `k=2`，`InitTaskTiling(k+1)` 的 `colNum=0`、`useCoreNum=0`。
- STEP260 的 8 份现场输入 A 都是有限 FP32 192×192 tensor；Q 的第 128～191 列和 R 的对应区域出现非有限值。同一 A 的 CPU FP32/FP64 QR 均有限。
- v1 生命周期/zero-init/const 组合候选已被 8/8 rank 的 raw profile 确认实际执行，但首次 `192×192 FP32` 现场输入仍输出非有限值，因此 v1 不能作为修复版发布。
- 原源码为 `alphaBuf` 分配 `2 * UB_ALIGN_SIZE = 64` bytes。已根据官方 `Duplicate` API 合同确认其第三参数是元素数，而不是字节数；两处传入 `64` 意味着写 `64` 个 FP32，即 256 bytes，每处超出 `alphaBuf` 192 bytes。
- v2 已在生成的 candidate 中引入显式常量 `ALPHA_BUF_BYTES=64` 和 `ALPHA_BUF_FP32_ELEMENTS=16`，保持分配大小不变，仅将两处 `Duplicate` count 改为 16 个元素。
- v2 candidate SHA256 为 `c4eef5c1984c10953420a9f30b9361473e8f33e2ccf280eefd1d8398c0e199c1`。将 alpha 的常量、分配表达式和两处 count 精确反向后，SHA256 必须回到已审计 v1 `5a4d140b8a473c3a0446d9e225431ff9f8be5e9b9f7355c5a166920e1814105b`，作为“v2 相对 v1 仅有 alphaBuf 增量”的 fail-closed 门禁。
- v2 首个 `192×192 FP32` 设备验证仍产生非有限输出；源码复审确认 `CalcQForLARFB()` 把 `vLocal` 通过 MTE3 写入 GM workspace 后，未建立 MTE3→MTE2 依赖就把同一 GM 地址交给 Matmul A。
- v3 candidate SHA256 为 `fbfda044ef5a15f45a1c48a3818d3d3360aa9c54ff39a36a1cb00e43cc813b99`。精确移除新增的一组 Fetch/Set/Wait 后，SHA256 必须回到 v2 `c4eef5c1984c10953420a9f30b9361473e8f33e2ccf280eefd1d8398c0e199c1`。
- v3 concrete AIC 已在 8/8 rank 各命中一次，但 rank0 的有限 192×192 FP32 输入仍产生全非有限 Q 和部分非有限 R，v3 已被设备证伪，禁止进入训练。
- 源码复审确认 v3 的 active LARFB 核共享同一 `workspaceInGm[m*blockSize]` scratch，`UpdateAForLARFB()` 的 UB→GM→Matmul B 路径缺少 MTE3→MTE2 依赖，且 core0 完成 Q 写回前其他核可释放并继续进入后续阶段。
- v4 candidate SHA256 为 `2213dbae5027c179f09c8e3b43be51587ba22e3eb2a7546311048efc3844614b`。精确反向每核 scratch、UpdateA 事件和外层 `SyncAll()` 三项后，必须回到 v3 SHA256 `fbfda044ef5a15f45a1c48a3818d3d3360aa9c54ff39a36a1cb00e43cc813b99`。

### 2.2 已构建但仍未通过的边界

- v4 已在目标 CANN 8.3 环境完成隔离 OPC 编译和封包，候选 wheel SHA256 为 `4c158915bd5ae3fad4834a4f88028702d2d6fb534d69da45cd06f0b536f8dead`；构建过程未安装、未训练，installed/runtime inventory 前后闭合。这只证明“可编译、可封包”，不证明运行正确。
- v4 concrete AIC 已在8/8 rank各精确命中一次，其他 QrV2 引用为0；但 rank0 真实有限且未修改的192×192 FP32输入仍产生 Q nonfinite=36864、R nonfinite=16448。因此 v4 三项同步/所有权闭合已被设备证明不足以消除现场非有限输出。
- 192×192 上三角总元素数为 18528，首个 64×64 对角块上三角为 2080，两者之差恰为 16448。这与 v4 的 R nonfinite 计数完全一致，支持“首次 GEQRT 生成的 R00 有限，首次 LARFB 从后续列开始污染”的路径判断。
- 源码静态审计确认两个位置合同错配：`vtvMatmulObj` 声明 A 为 `VECIN`却在 v4 的 LARFB 第二乘传入 GM；`qaMatmulObj` 声明 `VECIN/VECIN/GM`，而两条调用路径的实参均为 `VECIN/GM/VECIN`。
- v5 本地生成候选 SHA256 为 `e6ccbb84b0e0dbdc026ecdc6b6e07936fbd659401e35c38f7e9eb974d99bc3b7`；只反向上述两项修改必须精确恢复 v4 SHA256 `2213dbae5027c179f09c8e3b43be51587ba22e3eb2a7546311048efc3844614b`。
- 上述证据曾将 v5 定义为优先级最高的根因候选；后续 STEP373 已证明其可在目标 CANN 8.3 编译封包，STEP374 已证明8/8 rank命中 concrete AIC，但该内核第一次真实调用即 timeout/trap，因而没有形成恢复 finite/reconstruction 的因果证明。
- 尚未证明修复版本能通过 30-step loss 和性能门禁。

因此，v4 和 v5 均不再作为可发布候选继续扩大测试。当前只允许先离线映射 STEP374 的 PC/MTE 证据，并分别审计 v5 的两项 delta——`vtv` 直接消费 `vLocal` 与 `qa` 的 `VECIN/GM/VECIN` 类型变更——以及既有同步/所有权逻辑；在形成新的单变量、可回退候选前，不进行新的 NPU 调用、shape 扫描或训练。

## 3. 原始文件保护

- 原始 ZIP 全程只读，不覆盖、不重命名。开始前记录大小和 SHA256。
- 将 ZIP 解压到新的版本化工作目录；所有修改只发生在副本。
- 修改前单独保存目标 `qr_v2.cpp` 的原始副本、相对路径、权限、大小和 SHA256。
- 远端 installed `mx_driving_cloud` 全程只读。构建和测试只使用新的 diagnostics 目录和完整 shadow package。
- 每轮运行前后复核 installed 的两份 config、源码、wrapper、kernel JSON 和 object SHA。
- 每次构建和测试使用唯一目录，不复用或删除历史失败目录。
- 未经用户另行授权，不执行远端 installed 包替换。

## 4. 历史源码修改（v1～v5 均不得直接执行）

### 4.1 延后 LocalTensor 释放

原始所有权顺序为：

```text
GEQRT 入队 T/V/A
  → LARFB 出队并释放 T/V/A
  → Process 的 core0 再读取 T/V 构造 Q
```

修改后顺序为：

```text
GEQRT 入队 T/V/A
  → LARFB 出队并完成自身计算
  → core0 使用仍有效的 T/V 构造并写回 Q
  → 每个 core 各自释放一次 T/V/A
```

具体修改：

1. `LARFB()` inactive 分支保留三次 `DeQue`，删除三次 `FreeTensor`，随后返回。
2. `LARFB()` active 分支保留全部 MatMul、event 和 GM 写回，删除函数末尾三次 `FreeTensor`。
3. 在 `Process()` 中，将以下释放放到 core0 的 `CalcQForLARFB(false)` 和 `colQGm` 写回完成之后、TSQRT 循环之前，并置于 `if (coreId == 0)` 外：

```cpp
tTQue.FreeTensor<DTYPE_A>(tLocal);
vTQue.FreeTensor<DTYPE_A>(vLocal);
aTQue.FreeTensor<DTYPE_A>(aLocal);
```

每个 core 都释放自己的队列对象。不得只让 core0 释放，也不得修改 TSQRT/SSRFB 自身的 LocalTensor 生命周期。

### 4.2 初始化 zero-work tiling

`InitBaseTiling()` 的五个局部计数变量全部初始化为 0：

```cpp
uint32_t useCoreNum{0};
uint32_t formerNum{0};
uint32_t formerRepeatNum{0};
uint32_t tailNum{0};
uint32_t tailRepeatNum{0};
```

`InitTaskTiling()` 的计数、repeat 和 offset 全部初始化为 0：

```cpp
uint32_t useCoreNum{0};
uint32_t formerNum{0};
uint32_t formerRepeatNum{0};
uint32_t tailNum{0};
uint32_t tailRepeatNum{0};
uint32_t repeatNum{0};
uint64_t offsetK{0};
uint64_t offsetI{0};
uint64_t offsetW{0};
```

该修改消除 zero-work 和非活动核字段中的未定义值，不改变有效核分支的赋值和计算。

### 4.3 CANN 8.3 const 兼容

- `KernelLinalgQrV2::Init()` 的 tiling 参数改为 `const QrV2TilingData *`。
- kernel 入口按值复制四个已确认可复制的 `TCubeTiling` 字段，再传给 MatMul `Init()`。
- `op.Init()` 继续接收原 `tilingData` 的只读地址。

禁止使用 `const_cast`，禁止复制整个布局和复制语义未确认的 `QrV2TilingData`。

### 4.4 纠正 alphaBuf Duplicate 元素数

`alphaBuf` 的分配保持 64 bytes，但将字节数与 FP32 元素数分开命名：

```cpp
constexpr uint32_t ALPHA_BUF_BYTES = 2 * UB_ALIGN_SIZE;
constexpr uint32_t ALPHA_BUF_FP32_ELEMENTS = ALPHA_BUF_BYTES / sizeof(float);
static_assert(ALPHA_BUF_BYTES == 64, "alphaBuf must remain 64 bytes");
static_assert(ALPHA_BUF_FP32_ELEMENTS == 16, "alphaBuf must contain 16 FP32 elements");
```

`InitBuffer(alphaBuf, ...)` 仍传 `ALPHA_BUF_BYTES`；`SlarfgGeqrt()` 和 `SLARFGTsqrt()` 中的两处 `Duplicate(alphaLocal, ...)` 第三参数都传 `ALPHA_BUF_FP32_ELEMENTS`。不增大 UB 分配，不改变后续 Reduce 公式。

### 4.5 CalcQForLARFB 使用每核独占 scratch

v3 的所有 active 核都把 `vLocal` 写入 `workspaceInGm[m*blockSize]`，不同核的 MTE3 写和 Matmul A 读会争用同一地址。v4 使用同一个局部 offset 驱动 `DataCopy` 与 `SetTensorA`：

```cpp
uint64_t calcQScratchOffset =
    m * blockSize + coreId * blockElement;
DataCopy(workspaceInGm[calcQScratchOffset], vLocal, blockElement);
// 保留 v3 的 MTE3_MTE2 Fetch/Set/Wait
vtvMatmulObj.SetTensorA(workspaceInGm[calcQScratchOffset]);
```

范围证明：`LARFB()` 只允许 `coreId < useCoreNum` 的核调用该路径，且 `useCoreNum <= colNum <= blockp`；`Process()` 的额外调用只在 core0。因此 `coreId < blockp`。scratch 末地址不超过 `m*blockSize + blockp*blockElement = 2*m*blockSize`，精确落在 `workspaceInGm` 已分配的 `[0, 2*m*blockSize)` 元素范围内。

### 4.6 补齐 UpdateAForLARFB 的 MTE3→MTE2 依赖

在 `DataCopy(workspaceInGm[offsetW], aLocal, blockElement)` 后、同一地址交给 `qaMatmulObj.SetTensorB()` 前，新增独立的 `FetchEventID/SetFlag/WaitFlag<HardEvent::MTE3_MTE2>`。事件 ID 使用独立局部变量，不复用 `CalcQForLARFB()` 或 LARFB 循环的事件变量。

### 4.7 core0 Q 写回与每核释放之间建立全核 barrier

`Process()` 中 core0 完成 `CalcQForLARFB(false)`、`DataCopy(colQGm, qLocal, ...)` 及 `MTE3_V` wait 后，所有核在 `if (coreId == 0)` 外无条件执行一次 `SyncAll()`；随后每核各自释放 T/V/A，再进入 TSQRT。`SyncAll()` 禁止放在 core0 条件内，也禁止放在任一 `FreeTensor` 后。

### 4.8 v5 Matmul tensor-position 最小修正

v5 只在 v4 字节上执行两项变更：

1. `CalcQForLARFB()` 的第二次 `vtvMatmulObj` 不再将 `vLocal` 搬到 GM scratch 后传入声明为 `VECIN` 的 A，而是直接执行 `SetTensorA(this->vLocal)`。因为不再存在该 Local→GM→Matmul 路径，同时删除仅服务于该 scratch 的 offset、`DataCopy` 和 MTE3→MTE2 事件。
2. `qaMatmulObj` 模板位置从 `VECIN/VECIN/GM` 改为 `VECIN/GM/VECIN`，分别对应现有 `qLocal`、`workspaceInGm`、`aLocal` 实参。LARFB 和 SSRFB 两条路径都必须通过同一位置审计。

不允许在 v5 同时修改 GEQRT ReduceSum scratch alias、V→S event、Householder 公式或其他推测项。此处原计划在 finite 失败后直接做 T/V/q/a probe；STEP374 实际在 Q/R 产生前发生 runtime timeout/trap，已覆盖该旧分支。当前必须先按第15节完成PC/源码映射，再决定单观察边界probe。

### 4.9 明确保持不变的内容

- Householder、GEQRT、LARFB、TSQRT、SSRFB 数学公式和计算顺序。
- FP32 数据类型、64×64 分块、MatMul tiling 和 workspace 合同。
- `mx_driving_cloud.linalg.qr(A) -> (Q, R)` API。
- wrapper 的 padding、裁剪和 `torch.triu()` 行为。
- 正式内核不得包含 STEP350 的 dump、R 下三角写回、逐调用 finite 检查或 v4 之外的额外同步。
- v4 不初始化 Q/R，不新增 `baseTilingInfos` 槽位，不改 QR 数学、shape、dtype 或其他路径。
- v5 不改 workspace 总量或其他路径；仅因直接消费 Local `vLocal` 而删除 v4 中该次乘法专用的 GM scratch 使用及事件。

## 5. 历史构建与封装（v5 已 runtime invalid）

### 5.1 源码补丁后置断言

补丁器必须先校验原源码 SHA256，并以单次匹配方式修改完整语义块。生成 candidate 后必须自动断言：

- `LARFB()` active/inactive 两分支中 `FreeTensor` 数均为 0；
- `Process()` 新增的全核 `SyncAll()` 位于 Q 写回完成事件之后、3 次释放之前，3 次释放位于 TSQRT 之前；barrier 与释放均不在 `coreId == 0` 内；
- SSRFB 原有 6 个释放点和 `akkLocal` 的 1 个释放点未变；
- 全文 `FreeTensor` 调用数从原版 13 变为 candidate 10；
- `alphaBuf` 分配精确为 64 bytes，两处 `Duplicate(alphaLocal, ...)` 精确使用 16 个 FP32 元素，旧 `2 * UB_ALIGN_SIZE` 元素 count 必须为 0；
- 作为历史中间产物，v4 的 `DataCopy` 与 `SetTensorA` 必须精确共用每核 `calcQScratchOffset`；作为最终生成产物，v5 必须将该 scratch/copy/event 整段唯一替换为 `SetTensorA(this->vLocal)`，不得同时保留两条路径；
- `UpdateAForLARFB()` 精确新增一组独立 `MTE3_MTE2` Fetch/Set/Wait，且严格位于 workspace `DataCopy` 与 Matmul B `SetTensorB` 之间；
- v4 相对 v3 精确新增一处全核 `SyncAll()`，且必须在 core0 条件外、所有 T/V/A 释放前；
- 精确反向每核 scratch、UpdateA 事件和外层 `SyncAll()` 后，SHA 必须回到 v3 `fbfda044…13b99`；
- v5 必须相对 v4 只有两项位置合同差异：`CalcQForLARFB()` 第二乘直接使用 `vLocal(VECIN)`，以及 `qaMatmulObj=VECIN/GM/VECIN`；全部 `SetTensorA/SetTensorB/IterateAll` 调用必须与声明位置一致；
- 精确反向 v5 两项修改后，SHA 必须回到 v4 `2213dbae5027c179f09c8e3b43be51587ba22e3eb2a7546311048efc3844614b`；
- `baseTilingInfos[...]` 引用序列必须与原源码完全一致；
- 不存在 `const_cast`、诊断 dump、额外 finite 分支或 fallback。

上述断言只证明补丁结构，不替代 CANN 编译和设备实测。

### 5.2 官方 OPC 编译

实施时新建固定脚本 `.codex-tools/build_qrv2_release.py`，以已经过目标环境验证的 `.codex-tools/step338_build_mx_qr_candidate_remote.py` 为原型，去除远程连接逻辑和 STEP338 命名。固定包内输入为：

- wrapper：`mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.py`；
- source：同目录 `qr_v2.cpp`；
- 构建根：`diagnostics/qrv2_release_build/<soc_key>/`；
- descriptor：1 个 input、2 个 output，每个 tensor 仅保留 `shape=[-2]`、`format=ND`、`dtype=float32`；不携带 `ori_shape`、`ori_format` 或 `range`。`op_type=QrV2`、`attrs=[]`、`bin_filename=QrV2_matmul_position_fix_v5`。

SoC 映射固定为：

| 包目录 | 产物来源 | 验证等级 |
|---|---|---|
| `ascend910_93` | OPC `--soc_version=Ascend910_9362` | 当前目标设备编译+设备实测 |
| `ascend910b` | 逐字复制已验证的 `ascend910_93` `.o/.json` | DAV_2201 目录别名；与源产物 SHA 必须相同 |

只对当前真实设备标识 `Ascend910_9362` 调用一次同版官方 OPC，完整命令合同为：

```bash
opc <qr_v2.py> \
  --input_param=<input_param.json> \
  --main_func=qr_v2 \
  --bin_filename=QrV2_matmul_position_fix_v5 \
  --output=<isolated_output_dir> \
  --debug_dir=<isolated_debug_dir> \
  --soc_version=Ascend910_9362 \
  --op_mode=dynamic \
  --simplified_key_mode=0 \
  --optional_input_mode=gen_placeholder \
  --optional_output_mode=gen_placeholder \
  --deterministic=false
```

编译子进程仅临时设置原包 OPP Python 路径和
`ASCEND_CUSTOM_OPP_PATH=<original_customize_vendor_root>`，使 QrV2 custom tiling registry 可被加载。禁止修改全局环境、安装依赖或覆盖 installed 包。

正式 `kernelName/binFileName` 使用 `QrV2_matmul_position_fix_v5`。OPC 产物必须是唯一非空 `.o/.json`；`supportInfo.opMode=dynamic`、`simplifiedKeyMode=0`，两条 simplified key 必须与原动态 QrV2 合同精确一致。CANN 8.3 JSON 可不含 `kernelList`，但顶层名称必须精确匹配，且 `.o` 中 NUL 分隔的 concrete entry 集合必须精确等于 `QrV2_matmul_position_fix_v5_0_mix_aic/aiv` 两项。

该别名策略不是猜测：当前原 wheel 的 `ascend910_93` 与 `ascend910b` `.o/.json` 已证明逐字相同；`Ascend910_93` 与 `ASCEND910B` 均映射到 `DAV_2201`。实测在本 CANN 8.3 环境把同一源码强行以 `--soc_version=Ascend910B` 编译会产生 API/架构错误且 OPC 错误返回 0，因此不得伪称“双 SoC 独立编译成功”。构建 manifest 必须记录 `canonical_opc`、`dav2201_alias_copy`、源/目标目录和相同 SHA。最终正确性仍由目标机 concrete AIC、数学、loss 和性能门禁证明。

### 5.3 封装

- 在修复包副本中分别更新 `ascend910_93` 和 `ascend910b` 的 `qr_v2.json`；
- config 只允许改变目标 `binInfo.jsonFilePath`，`simplifiedKey`、输入输出定义和其他算子条目不得变化；
- 两个 config 必须分别指向本目录中的 JSON/object pair；两份 pair 必须逐字相同，JSON 顶层名称和 object concrete AIC/AIV 必须通过 5.2 门禁；
- 重新打包为新文件名，原 ZIP 保留；
- 生成原始/修复 ZIP、源码、wrapper、config 和双 SoC 产物的 SHA256 manifest。

## 6. CPU 官方算子语义对照

### 6.1 输入、padding 和输出 shape 合同

生产 wrapper 保持不变。另建测试专用 direct-call harness，逐行复制 wrapper 的 padding/crop 规则，并在调用 `_C.qr` 之前保留：

```python
A_original_before = A.clone()
A_pad = pad_exactly_like_production(A)
A_pad_before = A_pad.clone()
Q_pad, R_pad = mx_driving_cloud._C.qr(A_pad)
```

每个 case 记录原始 A、`A_pad_before`、CPU 同规则 `A_pad_ref`的 shape/dtype/stride/numel/SHA256。`A_pad_before` 与 `A_pad_ref` 必须逐位一致，padding 区域全部为 0。调用后只强制原始 A 与 `A_original_before` 逐位一致。

QrV2 会把传入 `_C.qr` 的 padded 临时 tensor 作为工作数据写回，因此不得要求调用后 `A_pad` SHA 不变；只记录其 after SHA 用于观察。测试 padding 函数和生产 wrapper 必须通过 AST/源码合同校验，防止两份逻辑漂移。

公开 API 输出 shape 依现有 wrapper 固定：

| 分支 | 条件 | Q shape | R shape |
|---|---|---|---|
| torch reduced | `min(m,n) <= 80` | `(m,min(m,n))` | `(min(m,n),n)` |
| MX complete-like | `min(m,n) > 80` | `(m,m)` | `(m,n)` |

这个阈值两侧的 shape 差异是现有 API 合同，本补丁不改变它。

### 6.2 数学 oracle

对原始 `m×n` 输入，记 `L=ceil(max(m,n)/64)×64`，内核参考是 zero-pad 后的 `L×L` FP32 矩阵 `A0`.在 CPU 上从同一 `A0` 计算 FP32 complete QR `(Qc,Rc)` 和 FP64 complete QR 摘要。令：

```text
u       = finfo(float32).eps / 2
gamma_L = L*u / (1 - L*u)
tiny    = finfo(float32).tiny
Erec    = abs(Qm@Rm - A0)
Brec    = 10 * (abs(Qc@Rc - A0) + gamma_L * (abs(Qm)@abs(Rm))) + tiny
Eorth   = abs(Qm.T@Qm - I)
Borth   = 10 * (abs(Qc.T@Qc - I) + gamma_L * (abs(Qm.T)@abs(Qm))) + tiny
```

每个 case 的硬门禁：

- A/Q/R 全部有限；
- finite 早退摘要分别记录 `input_finite`、`q_finite`、`r_finite` 和三者各自的 `nonfinite_count`；总判据仍严格等于三者逻辑与，不改变门禁阈值。worker 将该不含 tensor 的标量摘要单独持久化为严格 JSON，并保留原始 traceback；
- `count(Erec > Brec) == 0`；
- `count(Eorth > Borth) == 0`；
- 公开 API 单独定义 CPU 参考：`min(m,n)<=80` 时使用 `torch.linalg.qr(A, mode="reduced")`，`min(m,n)>80` 时使用 `mode="complete"`，得到 `(Qref,Rref)`；FP32 作 component-wise 参考，FP64 同mode只记高精度摘要/投影参考。令 `q=Qpub.shape[1]`、`gamma_q=q*u/(1-q*u)`、`gamma_m=m*u/(1-m*u)`，定义 `Erec_pub=abs(Qpub@Rpub-A)`、`Brec_pub=10*(abs(Qref@Rref-A)+gamma_q*(abs(Qpub)@abs(Rpub)))+tiny`、`Eorth_pub=abs(Qpub.T@Qpub-I_q)`、`Borth_pub=10*(abs(Qref.T@Qref-I_q)+gamma_m*(abs(Qpub.T)@abs(Qpub)))+tiny`；要求 `count(Erec_pub>Brec_pub)==0` 且 `count(Eorth_pub>Borth_pub)==0`；
- 对数值满秩 case，令 `r=min(m,n)`，比较 `Qpub[:,:r]@Qpub[:,:r].T` 与 CPU complete QR 对应投影，两个方向的 `rel_F/rel_max` 都按本节后述 `tol` 通过；对 rank-deficient case，非列主元 QR 的前 r 列未必是数值值域基，不将该投影作硬门禁，仍以重构、正交和三角合同为准；
- wrapper 执行 `torch.triu` 后，R 的严格下三角逐位等于 0；
- 额外记录 FP64 归一化 Frobenius 重构误差和正交误差，但不用 raw Q 逐元素相等作总裁。

数值秩使用 CPU FP64 SVD 定义：

```text
rank = count(sigma_i > max(m,n) * eps64 * sigma_max)
```

只对满秩且相邻奇异值间隙大于 `100*eps32*sigma_max` 的稳定 case，对前 `min(m,n)` 列作符号对齐后的 raw-Q 辅助比较。rank-deficient 或 complete QR 的零空间补基允许任意正交旋转，不比较这些列的 raw Q；此时以重构、正交、三角和 SOAP 下游语义为准。

对 SOAP 真实 `power_iter`，另外比较同一输入的 `project → precondition → project_back → parameter_delta`。每个阶段 tensor `x` 同时计算：

```text
rel_F(x,ref)   = ||x-ref||_F / max(||ref||_F, tiny)
rel_max(x,ref) = max(abs(x-ref)) / max(max(abs(ref)), tiny)
```

在同一进程/输入上执行 3 次 torch-vs-torch 控制，取两两比较的 `rel_F/rel_max` 最大值为 `control_max`。对每个阶段令 `d` 为该阶段最大缩约维度，硬阈值为：

```text
tol = max(1e-6, 10*control_max, 10*(d*u/(1-d*u)))
```

修复 MX-vs-CPU 的 `rel_F` 和 `rel_max` 都必须 `<=tol`。若 `d*u>=1`、control 本身出现非有限值或三次控制输入/RNG 不同，则该 case INVALID。这个有非零数值底线的规则避免确定性 torch 控制包络为 0 时误要求 MX 逐位一致。

### 6.3 测试集合

核心必测 case：

- 阈值：`(80,81)`、`(81,80)`、`(81,81)`；
- 分块：`(127,127)`、`(128,128)`、`(129,129)`、`(191,191)`、`(192,192)`、`(193,193)`、`(256,256)`；
- 矩形：`(129,81)`、`(81,129)`、`(193,129)`、`(129,193)`、`(192,256)`、`(256,192)`；
- STEP260 的 8 份真实 192×192 A。

扩展真实 shape 均为方阵 `(n,n)`：96、120、160、192、220、256、352、440、512、768、1024、2560、5120。每个 case 固定 CPU generator `seed=0`，至少包含：

- identity；
- `randn` FP32；
- `randn * 1e-8` 低幅值；
- 用 seed=0 和 seed=1 的两份 CPU FP64 Gaussian 矩阵分别执行 complete QR 生成 `U/V`，再组合 `U[:,:k] @ diag(logspace(0,-6,k)) @ V[:,:k].T`并 cast FP32 的近病态矩阵；
- 令 `r=floor(min(m,n)/2)`，在 FP32 `m×n` 零矩阵的前 `r` 个对角位置写入 `logspace(0,-3,r)`，其余位置保持逐位为零，生成在 FP32 存储后仍精确 rank=`r` 的 rank-deficient 矩阵。不得使用 dense FP64 `U/V` 组合后 cast FP32 的方式冒充精确秩，因为舍入噪声会使 FP64 SVD 秩门禁观测为满秩。

对 2560/5120 只执行 identity 和 seed=0 确定性随机，避免无意义地放大 CPU FP64 成本。真实 SOAP case 只使用已存在且 SHA 锁定的 `step260_qr_bad_tensors/rank{0..7}_step10_ind0_192x192_BAD.pt`，不伪造其他“真实”输入。所有用例不得在 CPU/MX 两边各自重新生成输入。

由于现场出现过“单次冷调用通过、训练上下文失败”，不再把同一 A 短次数重复当核心证据；已有 STEP303 证明连续 512 次冷调用也可全部有限。状态化核心测试改为：

1. `96 → 192 → 256 → 192 → 512 → 192` 交替序列；
2. 在独立 NPU stream 上重复该序列，用 event 建立必要的数据依赖，禁止用全局逐调用 synchronize 掩盖时序问题；
3. 正式 30-step 训练中的完整 SOAP 调用序列，这才是训练上下文证据。

前两个序列必须在 original shadow 和 fixed shadow 中各执行一次，按 mode/rank/调用序号/shape/dtype/输入 SHA 一一闭合。序列中每一次调用都独立执行 finite、重构、正交、三角和 shape 门禁。

### 6.4 原版/修复版直接 A/B

在同一批固定输入上用 fresh process 分别执行原版和修复版，两边都必须通过各自的 concrete AIC 身份门禁。比较项为输入/shape/dtype 合同、finite mask、重构、正交、三角和 SOAP 下游语义；raw Q/R 只做辅助观察。对原版已经正确的 case，修复版不得使任一数学指标超出同一 oracle 阈值或 torch-vs-torch 控制包络；对原版非有限的 STEP260 case，修复版必须转为全有限并通过全部数学合同。

该 A/B 同样必须按 rank/调用序号/shape/dtype/输入 SHA 一一对齐。STEP260 的 original 异常与fixed 有限只能作为“固定输入的 NaN 消除”证据；最终还必须由 30-step 真实 SOAP 调用账本、全程 finite 和 loss 门禁闭环训练上下文。

## 7. 历史设备身份方案（新候选以第17节为准）

- 从 installed `mx_driving_cloud` 物理复制完整 shadow package，仅在 shadow 中合入正式修复产物。两份 config、两份 JSON 和两份 object 必须是 shadow 内普通文件，禁止 symlink 或 realpath 逃回 installed。
- 任何 Torch/MX import 前，`find_spec("mx_driving_cloud")` 必须指向 shadow `__init__.py`。import 后 `mx_driving_cloud.__file__`、`mx_driving_cloud._C.__file__`、`ops/linalg.py` 必须都在 shadow 内，关键扩展/OP API 的 SHA 与原包一致。
- `ASCEND_CUSTOM_OPP_PATH` 第一项必须是 shadow customize，随后仅允许必要的 base/cloud fallback；不允许 installed cloud 出现在 shadow 之前。
- 所有测试和训练只在完整名称为 `mapqr-leicheng` 的容器中运行。
- 固定 `ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`，验证 8 rank 与后 8 张物理卡一一对应。
- 8 个 rank 各自采集 profiler raw。`hash_dic` 只接受 LF-only `<uint64十进制>:<name>\n`；`task_track` 只接受 64-byte 整倍记录，每条在 `+40` 读 little-endian uint64 hash。每 rank 都必须实际引用 `QrV2_matmul_position_fix_v5_0_mix_aic` 至少 1 次。
- 任何被 task 引用、名称以 `QrV2_` 开头但不是期望 candidate AIC/AIV 的 entry 都使整轮失败；CSV 中通用 `QrV2` 只作辅助，不能证明二进制身份。
- wrapper 按 rank 记录不含 tensor 的调用账本：shape、dtype、分支判断、`_C.qr` 调用数。每 rank 必须满足 `count(min(m,n)>80) == count(branch=MX) == count(_C.qr)`，并且 `min(m,n)>80` 的 torch/CPU fallback 计数为 0。在身份专用 run 中，调用账本与 profiler 的 QrV2 task 数必须闭合；正式性能 run 则依 8.3/8.4 仅在计时前做单次身份 probe。任一适用闭合不成立均标记 INVALID。
- 身份、rank、设备或 installed SHA 任一门禁失败，整轮标记 INVALID，不使用数学或性能结果。

## 8. 训练与性能验收

### 8.1 基线锁定

- 当前 GPU loss oracle 文件为 `gpu去除随机性固定后loss.log`，SHA256=`004282affd3c94781cd34c761f82d010e74b62178ce29eec1270fc901a8e70ca`；解析 JSON `.codex-tools/gpu_loss_800.json` SHA256=`67b36f3dbb36ff50b2a2bf68062d2e1589e2f55cb94207505fdd504e380a8851`。
- 启动前生成 `gpu_oracle_manifest.json`，锁定 loss SHA、source commit、config、checkpoint、数据列表、seed、world size 和 step 编号映射。无法验证的字段必须标记为阻断，不得自行补齐。
- 性能基线必须使用同一实测中的原始 MX shadow，不用 CPU QR、torch QR 或旧日志代替。原始/修复两组使用相同输入序列和调用账本；原始出现非有限值时仍记录耗时，但不把它当正确性样本。

### 8.2 30-step 精度

单算子全部通过后才运行纯 MX 30-step：

- 使用相同 checkpoint、配置、数据顺序、seed 和 RNG 摘要；样本/RNG 不一致则 INVALID；
- 每步总 loss 按 `abs(NPU-GPU)/abs(GPU)` 比较，GPU loss 为 0 时要求 NPU loss 也为 0，否则失败；
- 30/30 必须不超过 2%；missing、duplicate、NaN、Inf 或任一步超限都失败；
- 使用 `.codex-tools/step340_loss_gate.py` 的固定解析语义，不用均值或少数 step 替代逐步门禁；
- 正式 loss run 仍执行每 rank 调用账本和 concrete AIC 身份门禁。

### 8.3 单算子性能

原始/修复必须使用 fresh process、相同输入和相同 stream。每次计时在调用前后各同步一次；计时区间内禁止 profiler、dump、finite 扫描、tensor 保存和额外日志。

每个 fresh process 先执行一次不计时的 eligible QR，开启 profiler 并完成 concrete AIC 身份门禁；然后关闭/finalize profiler，在同一进程、同一已加载 shadow 中完成 warmup 后再开始正式计时。计时阶段用调用账本确认路径不变，原版/修复使用完全相同的身份 probe 以抵消持久化开销。

- shape ≤1024：5 次 warmup + 20 次计时；
- shape >1024：1 次 warmup + 20 次计时；
- 每 shape 的 median 取排序后中位数；P95 固定为 nearest-rank，索引 `ceil(0.95*N)-1`；修复/原始的 median 比和 P95 比均不得超过暂定回归阈值；
- 真实频次权重固定为：`96:3, 120:1, 128:18, 160:1, 192:32, 220:4, 256:181, 352:1, 440:4, 512:43, 768:22, 1024:6, 2560:8, 5120:4`。“加权 median/P95”分别定义为 `sum(weight_shape * per_shape_metric) / sum(weight_shape)`，修复/原始的两个加权比均不得超过暂定回归阈值。

当前将 10% (`ratio<=1.10`) 写为工程建议的暂定上限，因为用户只明确要求“耗时不能偏差太多”，尚未确认精确百分比。实施性能验收前必须由用户确认该数值；未确认时不作最终 PASS 裁决，也不允许在看到结果后放宽。

### 8.4 训练耗时

用 fresh original MX 和 fixed MX 各跑相同 30-step。每个训练进程在 Iter1 计时前以单次 eligible QR 做 profiler 身份 probe，finalize profiler 后才进入训练计时；训练阶段仅保留调用账本。比较 Iter2–30 的累计耗时和 QR cadence step 的中位数，两者 fixed/original 都必须不超过用户在实施前确认的回归阈值。样本、RNG、调用账本或身份 probe 不一致时，训练耗时结果 INVALID；GPU 耗时只单独报告，不作为本次 MX 回归的分母。

- 若单算子数学全部通过但 step12 仍越界，只能判定“QrV2 单算子修复尚不足以满足训练门禁”，不能直接宣称是独立 SOAP 问题。此时固定沿 A → Q/R 等价语义 → project → project_back → 参数增量 → 下一次 forward 定位最早分叉，以证据再分类，禁止 QR fallback。

## 9. 交付与回滚

最终交付：

- 新文件名的修复 ZIP；
- 最小源码 diff；
- 原始与修复源码、ZIP、双 SoC `.o/.json` SHA manifest；
- CPU FP32/FP64 对照报告；
- concrete AIC 身份报告；
- 30-step loss 和性能报告。

正式功能只形成一个提交：

`【npu性能优化】<根据最终已证因的 release delta 生成具体标题>`

测试阶段的回滚只需停止 shadow 进程并移除其环境变量，因为 installed 包未被修改。若未来获得安装授权，安装前必须归档原 package 的权限、owner、symlink、扩展属性和 SHA，并完成独立恢复演练。

## 10. 历史发布裁决（已失效，不得发布 v1～v5）

只有以下五项全部通过才允许发布：

1. 源码 diff 仅包含已批准的 v1–v4 历史修改，以及 v5 两项 Matmul 存储位置合同修正；反向 v5 两项差异必须精确恢复 v4 SHA；
2. CPU FP32/FP64 数学合同与全部 shape 测试通过；
3. 后 8 卡 8 rank concrete AIC 身份通过；
4. 30-step loss 30/30 不超过 2%，全程无非有限值；
5. 修复版相对原 MX 的加权耗时回归不超过用户实施前确认的阈值（当前建议暂定 10%）。

STEP350/351 内部上下文抓取不再是发布前置条件。只有修复版仍产生非有限输出，才重新启用该诊断路径。

## 11. 审核状态与证据边界

2026-08-21 的三项早期独立审核覆盖 v1 生命周期/zero-work/const 方案、CPU oracle 与设备验收设计。随后设备证据已证明 v1 concrete AIC 虽在 8/8 rank 实际命中，但首个 `192×192 FP32` 输入仍非有限，因此早期“方案 PASS”不是 v2/v3 产物的发布结论。

v2 实现后的独立只读审核已发现并驱动关闭以下交接缺口：

- alpha `Duplicate` 门禁从“包含常量名”改为两条完整调用精确匹配，增加 count 表达式膨胀反例；
- v2 alpha 变更精确反向后必须回到已审计 v1 SHA，同时定向拒绝新 `SyncAll()` 和 `baseTilingInfos` 槽位；
- shadow 交接层不再只检查 kernel/object 名称，还精确检查 dynamic `shape=[-2]`、`simplifiedKeyMode=0`、两条 simplified key、可选 `kernelList` 身份以及 `binary_info_config.json` 的 QrV2 路由。

v3 独立只读审核确认算子增量满足以下合同：只在 `CalcQForLARFB()` 目标 UB→GM→Matmul A 路径新增一组 MTE3→MTE2 事件，精确反向回到 v2 SHA，`UpdateAForLARFB()`、`SyncAll()` 和 base slot 不变。审核同时要求 finite Q/R 但后续 FP32 诊断溢出时仍生成 JSON-safe 摘要，并要求失败目录、摘要、traceback 三类 I/O 故障都不得覆盖原始算子异常；这些属于证据可靠性门禁，不改变算子或数学阈值。

v3 已由 8/8 rank concrete AIC 证明确实执行，但有限输入仍产生非有限 Q/R，因此已停止在 finite 门禁，未进入训练。v4 本地静态测试与目标 CANN 8.3 隔离构建/封包已完成，发布 wheel 已用唯一 SHA 接入 STEP371 设备控制器。这仍不代表 concrete v4 AIC、设备 Q/R、30-step loss 或性能验收已 PASS；禁止把“已构建”表述为“已修复”，也禁止把旧 v3 wheel 当作 v4 证据。性能最大允许回归仍待用户锁定，当前 10% 只是建议值。

## 12. STEP374 时点的历史顺序与停止条件（已被第17节替代）

以下顺序是当前唯一准入路径；上一级未通过，不得扩大到下一级：

1. **原件与变更范围**：锁定原 ZIP/源码/v4 SHA，只在新副本中生成 v5；反向两项 v5 变更必须精确恢复 v4 SHA。本地候选已通过。
2. **静态与构建**：v5 补丁结构断言、Python 测试、独立 builder identity 接线、目标 CANN OPC 编译、新 wheel 封包和 installed/runtime 库存闭合均已通过。
3. **首个真实输入核心门禁**：只在 `mapqr-leicheng`，只用物理后 8 卡，8 rank 各执行一次各自 STEP260 `192×192 FP32` 输入。必须同时满足 8/8 concrete v5 AIC 各精确命中一次、A 有限且未改，且每个 rank 的 shape、Q/R finite、reconstruction 和 orthogonality 都按第 6.2 节既定 oracle 硬门禁 PASS。重构/正交摘要只是证据载体，不是代替 PASS 谓词的诊断字段；installed 库存、进程/端口/后8卡清场也必须闭合。任一项失败立即停止，不跑 shape 全集或训练。
4. **算子语义门禁**：首输入通过后，再执行阈值、分块、矩形、病态、秩亏和真实 shape 集合；对同一输入按生产 padding/crop 合同与 CPU FP32/FP64 官方 QR 对照。硬门禁是 finite、shape/dtype、重构、正交、R 下三角和满秩子空间；不以 raw Q 逐元素相等作裁决。
5. **状态化上下文门禁**：原版/修复版在 fresh process 中执行交替 shape 和独立 stream 序列，不用每调用全局 synchronize 掩盖时序。按 rank/call/shape/dtype/输入 SHA 闭合账本。
6. **30-step 训练 loss 门禁**：只有前五级全通过才启动。同 checkpoint、config、数据序、seed、world size 和 step 映射；每一步 `abs(NPU-GPU)/abs(GPU) <= 2%`，30/30 全通过，missing/duplicate/NaN/Inf 任一出现即失败。不用均值替代逐步裁决。
7. **耗时门禁**：同时比较 fresh-process 单算子 median/P95 和训练 Iter2–30 累计耗时/QR cadence 中位数；original/fixed 的输入、stream、warmup、身份 probe 和调用账本必须相同。用户尚未确认“不能偏差太多”的精确百分比；在锁定前，10% 仅作预先建议上限，不得据测试结果事后放宽，也不得做最终 PASS 结论。
8. **交付/回滚**：只在 1–7 全部通过后生成正式交付结论。验收前始终使用 shadow，不替换 installed；任一失败通过停止 shadow 进程和移除本轮环境变量回退，原 ZIP 和 installed 不需要恢复。

当前执行位置：第 1、2 级已闭合；第 3 级唯一一次 STEP374 中，8/8 rank 的设备错误均点名 v5 concrete AIC，证明内核进入执行，但 profiler identity门禁未产出；第一次192调用均发生 `507014 / AclrtSynchronizeDeviceWithTimeout`，0/8 形成 done，无法获得 Q/R finite、reconstruction 或 orthogonality 结果。当前停止在内核执行故障定界，不得复跑、进入 CPU 语义、训练或性能门禁，也不得声称 v5 已修复。

## 13. 2026-08-21 独立审核结论

审核总结：P0=0。C++ v4 三项同步/所有权修复、精确反向 v3 SHA、不改 QR 数学/公开 API 的边界和 STEP370 发布物结构未发现 P0/P1。STEP370 的 wheel/外层 ZIP/RECORD、双 SoC JSON/O、dynamic simplified key、v4 concrete AIC/AIV、binary-info 精确 QrV2 delta 及 installed/runtime inventory 原位审计全部通过；证据等级仍为 `packaged_unvalidated`。

设备测试前阻断项：

- **P1：launcher 异常清场不闭合（已关闭）**。`.codex-tools/step358_host_case.py` 现在对 `Popen` 后 identity、`getpgid`、ownership 写入和 worker wait 异常统一保留主错误，然后无条件依次尝试 terminate、log close 和 `cleanup_owned_and_postflight()`；后续清场/证据写入失败只追加诊断，不覆盖主错误。三类前置失败及多重清场失败注入测试通过，独立复审 P0/P1=0。

正式训练前阻断项：

- **P1：loss 零基线语义不一致（已关闭）**。`.codex-tools/step340_loss_gate.py` 现已按本方案实现 GPU=0/NPU=0 PASS 且偏差为0，GPU=0/NPU≠0 以 `gpu_zero_npu_nonzero` FAIL；missing/duplicate/nonfinite 与非零基线的相对偏差语义保持不变。0/0、0/nonzero 及原回归测试共9/9 PASS，Python 审查 P0/P1/P2=0。
- **P1：验收总控尚未实现**。当前已有 loss 比较器，但尚无同时锁定 GPU oracle manifest、checkpoint/config/数据序/RNG、8-rank concrete AIC 和调用账本的 30-step 总控，也尚无 v5 单算子/训练耗时执行与聚合器。这不阻断 v5 的隔离 OPC 构建和首输入核心门禁，但阻断 30-step 启动与最终发布。

非阻断加固项：

- 将 build/runtime installed inventory 扩展到 C++ source、wrapper 和 `binary_info_config.json`；
- 对动态加载的 legacy controller 和 `remote_exec.py` 增加 SHA 门禁；
- 为 OPC `subprocess.run()` 增加有界 timeout。

本节的 v4 审核结论已被 STEP371 真机 finite 失败覆盖，仅作历史记录；不再授权任何 v4 构建或上机。v5 的当前审核裁决以下节为准。

## 14. 2026-08-21 v5 实施方案审核（STEP372时点历史裁决）

本节记录 STEP374 之前的准入判断，所有“当前”“未验证”和失败分支均已被第15节真机结果覆盖，不再授权构建或上机。

- **已通过**：原源码、v4 patcher 和原 ZIP 未修改；v5 候选仅有两项 Matmul 位置合同 delta；全部四个 Matmul 对象的 A/B/C 声明与现有实参静态一致；v5 精确反向恢复 v4 SHA；7/7 正反例、`py_compile`、`--check`、新 Python 文件 diff-check 和本次文档新增区段行尾检查通过。
- **已确认的修复目标**：消除 `vtvMatmulObj` 把 GM 实参交给 `VECIN` A 的错配，并使 `qaMatmulObj` 的 `VECIN/GM/VECIN` 声明与 LARFB/SSRFB 现有调用一致。
- **当时的合理推测**：上述错配位于首次 LARFB 污染路径，且 R 的 16448 个非有限元素与排除首个 64×64 R00 后的完整上三角区域精确一致，因此在 STEP372 时点被列为优先级最高的根因候选。
- **当时未验证**：目标 CANN 8.3 OPC、v5 concrete AIC运行及同一真实 192×192 输入的数学门禁。后续已确认OPC成功且设备错误8/8点名v5 AIC，但内核runtime timeout/trap，未得到Q/R；CPU语义、状态序列、30-step loss和耗时仍未执行。
- **阻断规则**：builder、shadow、worker、controller 必须全部锁定 `QrV2_matmul_position_fix_v5` 及新产物 SHA，不允许沿用 v4 wheel。只有当本地生成链、OPC 产物和 installed inventory 审计全部通过后，才允许一次真机核心门禁。
- **失败分支**：若编译失败，只分析类型/API 合同，不改环境版本；若 concrete identity 或 finite/reconstruction 失败，保留原位证据并转 T/V/q/a 阶段 probe，不扫 shape、不训练、不盲目重跑。

审核裁决：**v5 的修复边界和分级门禁方向可接受，但当前仅准入本地 builder/identity 接线与独立 OPC 构建；不准入训练，不得宣称根因已完整证明或算子已修复。**

## 15. STEP374 真机结果与后续审核边界

- **已确认**：STEP373 v5 wheel/RECORD、双 SoC 产物、静态 concrete identity 和 installed/runtime 库存审计闭合；STEP374 的8个 rank 均通过后8卡、world-size=8、shadow/OPP 和固定真实192输入前置门禁，并在设备错误中明确点名 `QrV2_matmul_position_fix_v5_0_mix_aic`，证明该内核进入执行；profiler identity门禁未产出、未PASS。
- **运行失败**：8/8 rank 在第一次调用的 `torch.npu.synchronize()` 阶段出现错误码507014、AICore timeout/trap及MTE错误信息；ready=8、done=0、failure traceback=8，未生成 failure scalar JSON 或 profiler identity JSON。
- **数值结论**：没有 Q/R 回传，不能判定 finite、重构或正交，也不能把本次现象表述为“NaN仍存在”。能够确认的是当前 v5 内核不能正常完成执行，因此该候选不具备后续门禁资格。
- **根因边界**：现有证据只把问题收敛到 v5 内核内部的同步、MTE访问或流水活性层；尚不能唯一映射到某条 AscendC 源码。尤其不能在没有PC映射或阶段证据时断言 `vtv` 直接消费 `vLocal`、`qa` 的位置类型变更、既有同步/所有权缺陷或它们与编译器 lowering 的交互中的任一项就是唯一原因。
- **清场**：postflight/finally PASS，端口34359无监听，后8卡无本轮进程，launcher加8 rank共9个受管PID/starttime均已退出；原始诊断和profile按规则保留。

审核裁决：**v5 已被真机执行门禁判为 runtime invalid。下一步只准入不运行NPU的源码/PC映射审计，以及审核通过后的单变量阶段诊断；禁止盲目复跑、shape扫描、CPU语义、30-step loss和性能测试。**

下一阶段的“单变量”必须满足：PC/错误地址先锁定一个调用点；每个 probe 只增加一个观察边界或一个可逆 delta；使用独立 kernel identity 与 SHA；明确 workspace、tiling、event 和同步增量。probe 只用于定位阶段，不自动升格为修复。只有由此形成唯一新候选、完成新身份/新SHA的构建与审计，并重新通过一次新的首输入 identity+数学门禁后，才允许恢复 CPU 语义、30-step loss 和性能阶段。

## 16. STEP375 PC映射与首个正交诊断候选

### 16.1 PC映射结论

- STEP373 v5 ELF的AIC符号范围为`0..0xe134`，AIV符号从`0xe134`开始；STEP374运行时AIC/AIV `pc_start`差值精确为`0xe134`，因此PC到kernel函数及相对offset的映射证据充分。
- AIC近入口簇为`+0x1368/+0x137c`共6个rank，深部簇为`+0xb838/+0xb848`共2个rank；AIV异常出现在4个rank且均为函数内`+0x8a4c`。这些位置是多核超时时的停留点，不自动等于首个故障指令。
- ELF不含DWARF或`.debug_line`，也没有`CalcQForLARFB`、`UpdateAForLARFB`等内部符号；现有OPC目录没有map/asm/line-info。因此本轮不能把offset映射到某一条AscendC源码，禁止用PC猜测具体`SetTensor`行。

### 16.2 两项delta的证据等级

- delta1（第二次vtv直接消费`vLocal`）：队列容量、对齐、生命周期、core0作用域和释放顺序没有发现确定性错误；SSRFB已有LocalTensor作为第二乘A的同类结构。它仍可能参与流水等待，但当前不是优先级更高的静态嫌疑。
- delta2（`qa=VECIN/GM/VECIN`）：同时改变qa的B读取与C写回lowering。两条GM-B路径已有MTE3→MTE2先行关系，但源码不能证明`IterateAll(aLocal)`后的CUBE/FIX→UB→MTE3完成关系；结合MTE timeout，它是活跃嫌疑，但尚未证明为根因。

### 16.3 STEP375 delta1-only诊断合同

相对v5只撤回delta2，保留delta1，生成独立诊断identity `QrV2_vtv_direct_qa_legacy_probe_v6`。源码SHA必须为`ef5db14e09170806acb7c5227fd619f3f5ffdc7d31f36e49058cc88987fce180`；反向delta1必须恢复v4 SHA `2213dbae5027c179f09c8e3b43be51587ba22e3eb2a7546311048efc3844614b`。qa必须精确保留legacy `VECIN/VECIN/GM`，报告必须为`diagnostic_only=true`和`release_candidate=false`。

该候选只回答“v5的qa delta2是否是runtime trap的必要条件”：

- 若仍以同类签名trap，delta2不是充分原因，继续审计delta1或交互；
- 若trap消失，只能说明delta2或其交互触发执行故障，不能说明legacy qa声明语义正确，更不能说明NaN已修复；
- 无论结果如何，STEP375不能直接进入CPU语义、loss或性能门禁。

STEP376诊断构建链已在本地实现并通过主流程测试与独立代码终审：adapter从结构上毒化`package/all`入口，只允许`prepare/build`；工作目录必须精确为approved root下的`work`；manifest从首次落盘即标记`diagnostic_only=true`、`release_candidate=false`、`package_forbidden=true`，构建后状态只能是`diagnostic_built_unvalidated`；seal前后均复验双SoC object/JSON/log、identity、大小和SHA闭包。专用controller只接受精确6个普通上传文件，独立核验构建前后installed/runtime/相关进程快照，并严格验证summary路径与SHA。

此前两个可观察性P2均已关闭：失败证据使用版本化、长度受限、严格校验的单行JSON，主异常与附加清理异常均不会发生日志注入或相互覆盖；所有远端Python入口统一`PYTHONDONTWRITEBYTECODE=1`。首次phase-transition后的唯一远端执行在exclusive目录shell检查失败，错误为`[: missing ']'`；严格遵守失败不重试。

只读现场审计确认正确诊断路径匹配数为0、OPC/adapter相关进程为0，installed/runtime文件mtime和摘要均早于本次失败，因此本次未创建目录、未上传、未执行OPC、未触碰安装态。根因是旧拼接在`shlex.quote(remote_diag)`与闭合`]`之间缺少空格，可精确复现同一rc=2与stderr。

controller通过独立`_exclusive_directory_script()`修复：`[ ! -e <quoted> ] || exit 73`，随后`mkdir -m 700 -- <quoted>`。bash语法、特殊字符路径、重复路径、悬空symlink、并发竞争及execute级失败不上传/不重试测试均闭合，46/46 PASS。

retry2 phase-transition允许使用全新诊断目录重新武装。当前DIAG_NAME为`step376_retry2_qrv2_vtv_direct_qa_legacy_probe_v6_build_20260822`，与首次失败名称明确不同；`BUILD_READY=True`。candidate identity/source SHA、五项上传输入SHA、adapter行为和禁止权限均未变化。新的controller/test SHA分别为`8160238cf1783a6754246cbc43bd1c9396250aeaaa95cd8049ff36ceecadc7e3`与`c10009796b6635557a86638bd151f53a647f84fe6e6d34541aa5af36b0862fa0`。截至本段落盘尚未执行attempt2，必须先完成armed-state终审；通过后仍只允许一次，失败不重试，且`package/wheel/install/modify-installed/NPU/train`始终禁止。

retry2实际在首个before-snapshot因OPC alias是symlink而停止，未启动OPC。修复后controller只从已验证container contract读取resolved `opc.path`，结构层拒绝缺失、非字符串、空值、NUL和相对路径；pre/post snapshot及container build统一使用该锁定realpath。legacy alias即使改指也不会改变执行路径，target缺失、symlink、path或SHA漂移仍由base builder fail-closed。base/adapter门禁未修改。当前重新`BUILD_READY=False`，49/49及OPC SHA定向1/1通过，独立审核P0/P1/P2=0；下一次远端尝试仍需全新phase-transition与DIAG_NAME。

attempt3 phase-transition已通过并以全新DIAG_NAME `step376_attempt3_qrv2_vtv_direct_qa_legacy_probe_v6_build_20260822`完成本地武装，`BUILD_READY=True`。测试精确锁定新名并拒绝attempt1/retry2历史名；candidate identity/source SHA、五项输入SHA、adapter/base与禁止权限均未变化。controller/test SHA分别为`426b9a458592ab47ecb8dccd15dad1b683207bd277b541e2615a08426020a1d3`和`f10793a40256e11c4eead98d4c67f512b9429c5d662f10132b7dacf3f6863ae4`，49/49加OPC SHA定向1/1通过。截至本段落盘尚未执行attempt3，必须先完成armed-state终审；通过后仅执行一次且失败不重试。

attempt3 armed-state终审P0/P1/P2=0后，唯一OPC构建成功并封存为`diagnostic_built_unvalidated`。双SoC object SHA均为`a75ff58a2e13d01eb7d8e3d04183478e1669ca83ed4e965993163c17dbb1be14`，JSON SHA均为`c34a02cbd02665ccd3563ec51d9ff92558ce28d7ec7fb2b54b224a0b3146c490`，bytes一致；每个SoC仅有独立诊断identity的AIC/AIV concrete entries。policy/candidate均为`diagnostic_only=true`、`release_candidate=false`、`package_forbidden=true`，package status为`forbidden_diagnostic_probe`，release wheel/zip不存在，installed/runtime前后闭合。

构建后controller已立即回退`BUILD_READY=False`。使用同一锁定helper环境再次原位重算manifest、artifact SHA、identity、release absence和当前snapshot，结果全部PASS，相关构建进程0；未下载、未安装、未运行NPU/训练。当前产物只证明诊断源码可在目标CANN编译并完成交接闭包，不证明runtime trap消失，更不证明NaN已修复。下一阶段必须先设计不打包、不改installed的诊断加载链并独立审核，随后才可能申请一次后8卡NPU诊断。

### 16.4 STEP377 diagnostic shadow与NPU诊断加载设计

诊断加载不生成新wheel：从attempt3 manifest锁定的immutable原wheel安全解包到全新O_EXCL shadow，只在shadow内对两个SoC删除旧QrV2 object/JSON、复制诊断object/JSON，并唯一更新对应`qr_v2.json`和`binary_info_config.json`路由。不得overlay source、改RECORD、调用package/release/install、生成wheel/zip或写installed。原wheel、attempt3 artifacts、installed/runtime在前后均需SHA/mtime闭包。

首轮runtime只允许后8卡8-rank、每rank一个固定STEP260 `192×192 FP32`真实输入、一次profiled `_C.qr`。每rank必须证明task-referenced concrete诊断AIC恰好1次，旧installed/v4/v5 identity为0；输入identity/shape/dtype/bitwise不变、Q/R finite、重构/正交/投影等既有oracle谓词全部PASS。任何失败立即清场并停止，不扩大shape、状态序列、训练或性能。

专用controller初始必须`NPU_READY=False`，在读取映射或连接前fail-closed；后8物理die与8个rank PID/NSpid双射、端口/ownership/starttime/PGID和finally清场必须闭合。runtime结论只能是`diagnostic_runtime_pass/fail`及对delta2必要性的有限判断，不能写fixed/release-ready。当前仅准入本地diagnostic shadow builder实现与测试，尚未授权NPU。

STEP377 shadow builder现已实现并终审P0/P1/P2=0。它逐键接受真实STEP376 full policy并锁定诊断flags、candidate identity/source/reverse-v4与四工具SHA；ZIP采用member/ratio/声明与实际累计上限的流式解压；原树与结果树做完整path/type/mode/size/SHA差分，只允许双SoC旧pair删除、新pair新增及四个配置文件修改。overlay全程使用dirfd、`O_NOFOLLOW`和`*at`操作；`shadow.partial`携带不可消费marker，完成树复验并删除marker后最后O_EXCL发布manifest，manifest是唯一完成信号。官方wheel必须恰有一个锁定RECORD，所有输入前后SHA闭合。最终builder/test SHA为`bb080a8209f55327bd5774ac14ae99d5d58ba9ff8147140a447b9f37ab6356ff`和`1c804276717fe5fcbeb3cdc1612aba048cfc0b05a7b5d280dcaa7bab88e0a938`，16/16通过。当前仍未接真实远端产物或NPU。

STEP377 diagnostic worker采用薄适配而非复制数学实现：SHA锁STEP358 worker、cold-case和oracle，CLI不暴露多case选项并向底层恰好追加一次`--first-profiled-only`。profiler先走原验证，再按identity聚合task references；同identity多hash直接拒绝，全集只能是diagnostic AIC=1，AIV/original/v4/v5/unknown均为0。release等待被映射为全新diagnostic gate，等待前必须不存在，返回后以lstat→O_NOFOLLOW open/fstat→lstat锁regular与inode。异常路径对sys.modules、identity、wait和argv逐项恢复，主异常优先。最终adapter/test SHA为`1be60dd31e233c3e73f64dbad6747edad0d92201c576decda2c628737bb96519`和`16d1a891545ab6dda7e5864f8846386bdee5fbec34a50c1ac2a86c9423e6ce53`，12/12通过，终审P0/P1/P2=0。尚未运行torch/NPU。

诊断构建闭包通过后，是否允许一次新的后8卡NPU诊断仍需单独阶段审核。其结果无论是trap保留还是消失，都只能更新delta2必要性判断，不能自动升格为修复。只有据此另行形成正式修复候选后，才能以全新identity/SHA依次重走构建、首输入profiler identity与数学、CPU官方QR语义、状态序列、30-step逐步loss≤2%和耗时门禁。

### 16.5 正式修复实施路径与 STEP377 审核裁决（2026-08-22）

正式修复不直接采用 STEP375/376/377 的诊断产物。执行顺序固定为：先用 delta1-only 诊断回答 v5 `qa` delta2 是否为 runtime trap 的必要条件；再依据唯一设备证据修改包内相对路径 `mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.cpp` 的新副本，生成全新 release identity/SHA；随后依次通过同一输入 CPU FP32/FP64 官方 QR 的 shape、padding/crop、finite、重构、正交、R 下三角与符号/子空间语义门禁，后8卡8-rank concrete AIC 首输入门禁，状态化调用门禁，30-step 每步 loss 相对 GPU 日志 `<=2%` 门禁，以及测试前锁定的耗时门禁。任一级失败即停止，不能用后级均值掩盖前级失败，也不能修改原 ZIP、原源码或 installed 包。

STEP377 controller 文件闭包实现的首次独立审核裁决为 **不通过，P0=2、P1=3**，因此 `NPU_READY=False` 必须保持：route 校验仍允许 `..` 路径穿越；shadow artifact/RECORD 尚未精确绑定两个 SoC 包内固定路径、原 wheel dist-info 与完整树；upload inventory 缺读取后的目录项复核，本地上传源存在 hash 后再次读取的替换窗口；installed inventory 尚未拒绝额外 SoC、额外 `qr_v2` 目录和其他位置的 QrV2 route。安全清场还必须另行解决 PID/PGID 复用 TOCTOU、双次稳定 `npu-smi` 采样和 PID starttime/NSpid/host/container/device/rank 严格双射。

只有这些 P0/P1 全部修复、相应 race/路径穿越/额外路由负例通过，并再次独立审核达到 P0=0、P1=0 后，才能另行评审一次性 NPU 诊断；该诊断即使通过也不是正式修复验收。

## 17. 当前可执行实现方案（2026-08-22）

### 17.1 方案裁决

本节是当前执行入口；第 4～16 节中的 v1～v5 及 STEP375～377 只作历史证据，不再单独授权发布或训练。

- **已确认事实**：原始 QrV2 对有限 `192×192 FP32` 现场输入产生最后 64 列非有限；v1～v4 未修复该问题；v5 首次调用发生 runtime trap；CPU FP32/FP64 官方 QR 对同一输入正常。
- **合理推测**：LocalTensor 生命周期、zero-work tiling、Matmul tensor-position 与事件/同步合同都可能参与故障，但任一项都尚不是经设备单变量闭环证明的唯一根因。
- **未验证假设**：STEP377 delta1-only 能否正常完成调用，`qa` delta2 是否为 v5 trap 的必要条件，以及最终哪个最小 delta 能同时消除 nonfinite 且保持语义和性能。

因此，不预先锁定“延后释放 + zero-work + alphaBuf + 全部 event/barrier + v5 位置变更”的叠加补丁。当前实现采用证据驱动的最小必要 delta 集：诊断时每轮只引入一个可逆原子 delta，发布前对集合逐项消融；产物使用唯一 identity/SHA，任一核心门禁失败即拒绝该候选。

### 17.2 实现顺序

1. **诊断闭环，不作发布物**：STEP377 实验链的 CLI/清场缺陷已完成本地修正和审核。attempt6 暴露单元测试逃逸及 installed OPP 路径硬编码，attempt7 暴露 `NSpid` Tab 解析错误，attempt8 暴露 ready `wrapper_contract` schema 与真实 worker 不一致；三者均未进入 QrV2 数值证据阶段，已退役且不得复用。attempt9是唯一次后8卡、8-rank、每rank一个现场输入的delta1-only诊断：8/8 ready/ownership/gate ack，profile含尚未闭合rank覆盖和concrete AIC身份的generic `QrV2` marker，但done=0、summary=0、Q/R/finite字段=0，因而只能裁决“delta1-only不能形成可发布的正常返回证据”，不能证明8/8进入concrete kernel、定位终止阶段，或确定`qa` delta2是否为trap必要条件。controller已立即重新锁定`NPU_READY=False`，attempt9不得重跑；终态只读双样本审计确认exact case、端口和后8卡进程均0。
2. **选择最小必要正式 delta 集**：STEP377 delta1-only 只能判断 `qa` delta2 是否为 v5 trap 的必要条件，不能单独证明原始 nonfinite 的修复。若它未同时形成“同一输入→唯一阶段→输出”的因果证据，则只允许继续事前定义、每次一个可逆原子 delta 的诊断。若一个原子 delta 足以修复，其在同一现场输入上必须正常完成且 Q/R finite、重构/正交/三角全部 PASS，反向它必须恢复故障。若必须组合多个原子 delta，则完整集合必须正向通过同一门禁，并逐项移除每个 delta：移除任一必要项都必须恢复对应故障或使硬门禁失败，不影响结果的项必须删除。manifest 逐项记录 delta、正向/消融证据和反向结果；整个功能仍以一个可回退 commit 交付。仅修改原 ZIP 解压副本中的 `mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/qr_v2.cpp`；原 ZIP、原源码镜像和 installed package 保持不变。补丁器须先验证基线 SHA，且精确反向后恢复基线 SHA。
3. **静态与构建门禁**：限定变更文件/语义块，检查 LocalTensor 每核所有权、zero-work 定义、GM/UB 边界、Matmul A/B/C 位置、event 依赖、barrier 收敛和释放次数；只做隔离 OPC 构建和 shadow 封装，不覆盖 installed。
4. **CPU 官方 QR oracle 冻结**：对不得拉回本地的现场 tensor，CPU FP32/FP64 oracle 必须在远端原位、针对同一锁定输入生成。逐 case manifest 锁定 input SHA、CPU/PyTorch 版本、padding/crop/mode、reference/bound artifact SHA、阈值和判定字段，产物原位保留且不下载。校验输入 bitwise 未改、shape/dtype 一致，并固定 finite、`Q@R≈A`、`QᵀQ≈I`、R 下三角、满秩子空间/投影谓词。QR 存在符号和退化子空间不唯一性，不以 raw Q/R 逐元素相等作硬门禁。
5. **后 8 卡首输入与 CPU oracle 对照门禁**：只在完整名称 `mapqr-leicheng` 容器，使用物理后 8 张 Ascend NPU，验证 `torch_npu`、8 ranks、rank PID/NSpid↔physical die 双射、候选 concrete AIC 精确命中。8/8 的 A/Q/R 身份、finite 和数学谓词必须全部通过，并将实际 Q/R 与第4步 oracle 在远端原位逐 case 对照。每个 case 必须机械断言 input/padded-input/kernel-Q/kernel-R/published-Q/published-R 的 expected/actual shape、精确 FP32 dtype、device 和 layout 合同；任一失败即停止。
6. **状态化与 shape 门禁**：所有正式 shape 测试都继续满足同一容器 + 物理后8卡 + 8-rank + PID/NSpid/device 双射硬门禁。在 fresh process 中覆盖分支阈值、64 分块边界、矩形、病态、秩亏、交替 shape 和生产 stream/调用顺序；每次按 rank/call/shape/dtype/input SHA 闭合，不用每调用全局 synchronize 改变时序。
7. **30-step loss 硬门禁**：固定 checkpoint、config、数据序、seed、world size、step 映射以及 log 解析字段/单位/数值精度，比较同一 step 的总 loss 原始标量。对 GPU log 每步计算 `abs(NPU-GPU)/abs(GPU)`，30/30 每步必须 `<=2%`；GPU loss 为 0 时只有 NPU loss 也为 0 才通过。missing、duplicate、NaN、Inf 任一出现即失败，不用均值或最终步代替。该轮也必须闭合后8卡、8-rank 和物理映射。
8. **性能硬门禁**：本次修复的唯一性能硬门禁是 fixed/original MX 在同一 NPU 合同下的允许回归上限，须由用户在测试前锁定；当前 10% 只是建议，未确认前不得作最终 PASS。CPU FP64 SOAP 和 8-GPU 性能只在已存在完整、不可变 manifest 的同合同外部基线时作独立参考/项目目标，不在客户环境启动 CPU/GPU 正式训练，也不阻断本次修复发布，除非用户另行明确将其设为硬门禁并预给合法数据源与阈值。正式性能统计固定为 5 个 paired fresh-process 批次，执行顺序在测试前固定为 `O1,F1,F2,O2,O3,F3,F4,O4,O5,F5`，其中同编号 O/F 使用相同 warmup/输入/账本；bootstrap seed 固定为 `20260822`，重采样单位是 5 个 paired batch ratio，以 paired ratio 的 median 为中心统计量，重采样 10,000 次计算 95% CI。每个 O/F 运行从进程启动前到结束后，每 1 秒通过当前环境已有的 `npu-smi info` 读取后8卡频率/温度与进程；每卡以该运行首个有效采样为起始值，任一时点相对波动超过 5% 则该 paired batch INVALID，8 卡不做均值掩盖；任一卡任一预期采样缺失也为 INVALID。同时记录 host load 和输入/调用账本；出现非本轮 NPU 进程或账本不一致时，整批 INVALID。每个编号最多允许 1 次原位重试，必须同时废弃该编号原 O/F 两个结果，以全新 fresh process 按原顺序重跑整对；重试仍 INVALID 则整轮 INVALID，禁止新增批次、换序或挑选。温度和 host load 只作观察字段，不得事后用于选择性排除批次。对单算子 median、P95、30-step Iter2～30 累计耗时和 QR cadence 中位数分别计算 fixed/original ratio；每个指标的 paired-ratio median 点估计、95% CI 上界和最差 paired batch ratio 三者必须全部 `<=` 用户预锁上限。

### 17.3 发布与回退条件

只有 17.2 的 1～8 按顺序全部通过，且独立审核 P0=0、P1=0，才能生成待交付修复包。正式候选在 STEP377 证据完成前仅用 `RELEASE_ID/SOURCE_SHA/BASE_SHA` 占位，不得预填 v4/v5；发布 diff 只允许包含经证明的最小 delta，不默认继承 v1～v5 任一历史集合。发布物必须包含源码/产物 SHA manifest、唯一补丁、反向恢复真实 `BASE_SHA` 的机械校验、CPU/设备/状态/loss/性能报告和回退指令。任一失败候选保留为隔离诊断证据，不覆盖原件、不修改 installed、不进入后续门禁。

### 17.4 当前审核入口

本轮只审核方案，不执行算子或训练。审核项固定为：证据等级是否误标；诊断与 release 产物是否隔离；修改是否最小、可逆和可归因；CPU 官方 QR 对照是否处理不唯一性；后 8 卡、loss `<=2%`和耗时门禁是否可机械执行；原件/installed/回退闭环是否完整。

2026-08-22 独立只读审核（修订前）：`P0=0，P1=2`。P1 分别是“当前状态滞后，历史动作可能被重复执行”和“性能样本顺序、INVALID 重试及频率采样未完全机械化”。本次已在第17节更新 attempt6～9 的唯一执行指针，并冻结 O/F 顺序、采样接口/周期/逐卡裁决和单次整对重试规则。仍有三个明确阻断项：用户尚未锁定性能回归上限；GPU oracle manifest 尚未实际生成并验证；正式 release 的 `BASE_SHA/RELEASE_ID/SOURCE_SHA` 尚未产生。在这三项闭环前，本方案可用于诊断和候选实现，不得用于最终发布 PASS。

2026-08-22 修订后终审：`P0=0，P1=0，P2=0`。终审特别核对了attempt9的证据等级：generic `QrV2` marker没有被当作8/8 concrete AIC进入或底层根因证据。当前方案可继续用于诊断和正式候选实现；上述三个阻断项未闭环前，仍不得裁决最终发布PASS。
