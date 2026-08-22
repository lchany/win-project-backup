# Findings & Decisions

## 2026-08-22 MX QrV2 当前实现决策

- 不将静态可见的 LocalTensor 释放后读取直接标为已证明唯一根因：包含生命周期处理的 v1 仍被真机 nonfinite 门禁证伪。
- 不将 v5 的两项 Matmul position delta 当作正式修复：v5 已在首次真实 192 调用中 runtime trap，没有 Q/R 数学结果。
- 当前选择证据驱动、单变量、可逆的候选生成路径；STEP375～377 只用于区分 v5 trap 机制，不是 release 产物。
- 正式候选必须依次通过 CPU 官方 QR 语义、后8卡8-rank concrete AIC、状态化调用、30-step每步loss相对GPU `<=2%`和测试前锁定的耗时门禁。
- CPU 对照硬门禁是 shape/dtype/padding-crop/finite/重构/正交/三角/满秩子空间，不是 raw Q/R 逐元素相等。
- 性能回归上限仍未由用户数值化锁定；文档中 10% 只是建议，在确认前阻断最终 PASS。
- STEP379 新证据：attempt6 只证明运行链硬编码的 installed custom OPP 路径不存在，8/8 rank 均在导入torch/torch_npu、加载输入和QrV2调用前退出。不能用该结果判断delta2、runtime trap、nonfinite或修复正确性。
- attempt6 还证明pre-rank-evidence失败路径的cleanup语义不完整：全部worker已退出时应该通过，但无evidence的live worker不得获得signal权限，必须fail-closed。
- STEP380 新证据：attempt7已通过installed root/shadow/world8前置并产生8/8 ready，但Linux `NSpid` tab-separated chain被只接受普通空格的正则拒绝，导致release gate前失败。该轮仍没有QrV2调用证据，不改变delta2或nonfinite判断。

历史全文已备份到 `planning_backup/2026-08-18/`。本文件只保留当前仍会影响决策的结论。

## 2026-08-21 本地相关文件读取
- 根目录已有 `task_plan.md`、`findings.md`、`progress.md`，已按 `planning-with-files` 恢复上下文。
- `操作步骤.md` 约 1.24 MB / 7154 行；采用关键词索引与相关段落读取，不做无差别全文输出。
- `机器IP.md` 共 25 行，敏感字段只做脱敏解析；已确认主训练机地址末段为 `42`，但尚未发起远程连接或验证主机身份。
- GBrain bootstrap worker 未取得项目 ID：当前 MCP 无 `ensure_project`，本地 CLI 未配置 brain；未执行 bind、登记页验证或 cleanup。
- 项目 `AGENTS.md` 已完整读取：远端训练只能在精确名称 `mapqr-leicheng` 的现有容器内使用 8 张昇腾 NPU；远端依赖版本禁止变更；远端产物不得拉回本地；原始 profile 默认保留。
- `/home/l30002999/import-md/hw-import-ip.md` 已脱敏读取：它登记 4 台内部机器的公私网映射及若干中转机；内部机器之间优先走私网。该文件没有替代项目 `机器IP.md` 对目标主训练机末段 `42` 与双跳路径的约束。
- 用户随后明确收紧范围：后续不得读取当前项目目录外的任何文件。后续读取与结论仅使用 `/home/l30002999/project/MapQr/win-project-backup/DongFeng` 内文件；不再访问或依赖目录外材料扩展分析。
- 顶层文档索引已读取。当前状态应优先采用更新时间更晚的 `task_plan.md` / `findings.md` / `progress.md` / `操作步骤.md`；`PROJECT_STATUS.md` 标注更新时间为 2026-08-14，包含后来已被 STEP-299/301/303/331～334 修正或推进的旧状态，不能单独视为最新真相。
- `DrivingSDK优化研究与实施计划.md` 是候选研究流程与历史队列；其中部分“立即执行项”已被后续状态覆盖，使用前需对照当前计划和操作记录。
- `QR算子.md` 主要是分块 QR 方案说明和预期收益描述；其中“3–5 倍”等属于方案宣称，不等同于当前项目实测结论。当前实测以 STEP-315～334 及规划文件为准。
- 用户澄清：`planning-with-files` 只允许读取和维护当前目录内的规划相关文件，不得到其他目录读取该技能的脚本、模板、说明或其他材料。此前目录外读取发生在澄清前；后续范围严格限定为当前项目目录。

### 已确认事实
- 最新实验链应以 `操作步骤.md` STEP-315～334 为准：正式源码已在 STEP-327 改回 `torch.linalg.qr`（提交 `3a1d763`），30 步内消除了 mx QR 路径的 NaN；但这并未同时解决 GPU loss 双门禁和总耗时。
- STEP-330～334 已证实 iter4/14/24 的百秒长尾来自 QR 工作在设备时间轴上的兑现；STEP-332/333 显示 8 rank 的 `query_false=0`、`sync_ms=0`，所以显式 install synchronize 不是瓶颈。
- STEP-334 的 Call Stack 只发现两条 QR 触发路径：同步 `get_orthogonal_matrix_QR` fallback，以及 stale-Q submit/finish；没有 install 调用栈。
- 旧 STEP-270/285 的“npu2–7 固有失效”结论已被 STEP-299/301 纠正：旧 harness 缺少正确的 `torch.npu.set_device`。正确绑定设备后，受控单算子测试未复现 507015。
- 1:1 目标仍未在当前双门禁合同下完成。STEP-227 的约 0.97:1 属于较早、不同正确性合同的性能栈；STEP-326 的 torch QR 短期方案虽 finite，但总耗时约 GPU 4.16 倍且 loss 后段偏离，二者不能直接合并成“已达标”。

### 合理推测
- 下一性能突破更可能来自减少 QR 次数/大 shape、扩大 stale-Q 覆盖以消除同步 fallback，或改善侧流 QR 与前向的真实重叠；这些方向尚未形成通过双门禁的方案。

### 未验证假设
- `QR算子.md` 的分块实现能达到“3～5 倍”收益尚无当前合同下的独立实测支持。
- QrV2 最后一块 tile 的源码风险是否仍是训练态 NaN 的直接原因，在 STEP-315～325 新证据下尚未被最终闭环；isolated dump 张量测试均 finite。

## 2026-08-21 MX QR 根因验证重启

### 已确认的夹具缺口
- STEP-322 只检查两种实现各自 finite 与 `Q@R≈A`，没有检查 `QᵀQ≈I`、R 三角性、输入未修改、dtype/shape、stream/current-device 或重复调用稳定性。
- STEP-323 使用 `max(abs(Q_mx)-abs(Q_torch))` 和原始 R 差异判断实现不一致。QR 分解不唯一，这个逐元素判据不能区分合法的列符号、列旋转或近退化子空间基变换，也不能证明 MX 错误。
- STEP-319～323 每个 rank/shape 只挑第一份 dump；没有保持训练中的 QR 调用顺序、同进程 allocator/stream 状态、SOAP `sort_idx`、`exp_avg_sq` 与连续周期状态。因此 isolated 全 finite 不能否定训练上下文故障。
- 当前工作树由上级 Git 仓库观察为大量 tracked deletion + 当前目录重新未跟踪的异常状态；在厘清仓库身份前不得改业务代码或提交，只允许当前目录内只读诊断和规划记录。
- STEP-315 的 `_dump_one` 实际只保存输入 `A`，不保存 Q/R 或 SOAP 下游状态；文件名中的 `nonfinite_output` 只是一条触发标签。因此旧记录中“QR dump 输出 finite/nonfinite”的表述不能从 `.pt` 内容直接复核。
- STEP-315 hook 在每次 QR 后执行 `torch.isfinite(...).all().item()`，会强制设备到主机同步；它改变了原训练的异步时序，不能作为无扰动复现夹具。torch backend 模式还会经过 MX wrapper → patched torch wrapper 的嵌套路径。
- 本地 `qr_v2.cpp` 显示一个静态可证风险：最后一个 block 的 `Process()` 调用 `InitTaskTiling(k+1)` 得到 `colNum=0/useCoreNum=0`；`LARFB` 随后让所有核 `DeQue` 并 `FreeTensor(tLocal/vLocal/aLocal)`，但 core0 返回后仍调用 `CalcQForLARFB(false)`，继续读取 `tLocal/vLocal`。这是明确的释放后使用路径，能够解释状态/时序相关的未定义输出，但尚需验证该源码与远端实际加载二进制完全对应。
- `InitTaskTiling` 与 `InitBaseTiling` 在小于核数的分支没有初始化 `tailNum/tailRepeatNum`；当 `colNum=0` 时 core0 进入读取 `tailRepeatNum` 的分支。虽然 `LARFB` 会因 `useCoreNum=0` 早退，未初始化字段仍是第二个确定性源码缺陷信号。

### 当前最强根因候选
- 若能证明本地 `qr_v2.cpp` 与远端 MX wheel/加载 kernel 同版本同哈希，则“最后 block 的 local tensor 释放后继续使用 + 零列 tiling 未初始化”是目前最具体的算子级根因候选；它比原始 Q/R 逐元素差异更有证据力。

### STEP-260 原始失败张量重新核验
- 本地 8 个 rank 的 STEP-260 BAD `.pt` 都包含真实 `A/Q/R/meta`，区别于 STEP-315 只保存 A 的 dump。
- 8/8 输入 A 全有限；8/8 Q 各有恰好 `12288 = 192×64` 个非有限元素，坏列完整覆盖 128～191。
- 8/8 R 各有恰好 `10272 = sum(129..192)` 个非有限元素，精确覆盖上三角矩阵最后 64 列的全部有效位置；下三角由 Python wrapper 的 `torch.triu` 清零。
- 该计数与 64×64 tiling 的最后一个列块完全一致。这是直接的算子数学合同失败，不能由“QR 基不唯一”解释；同时与 `Process()` 最后 block 的空 LARFB / 释放后继续 CalcQ 静态路径强吻合。
- 仍缺最后一环：确认当前远端实际加载的 wrapper、`qr_v2.cpp` 和编译 kernel 与该本地快照同源，并用修复后的 kernel（不是 Python bypass）做关闭 bypass 的 8-NPU A/B。

### 远端 provenance 门禁（STEP-337）
- 42 主机身份守卫通过；精确容器 `mapqr-leicheng` 唯一且 Running。
- 容器环境：torch `2.7.1+cpu`、torch_npu `2.7.1`、NPU available、设备数 16；未改环境。
- 远端安装 `linalg.py` SHA-256 与本地 `linalg_official_26.0.7.py` 完全一致：`2e2171c4...02ee3`。
- 远端安装 `qr_v2.cpp` SHA-256 与本地快照完全一致：`2dbaf1e1...1206c9`；源码确认 `colNum=blockp-k`、core0 无条件 CalcQ、LARFB 释放 vLocal，且没有 `useCoreNum` guard。
- 实际包内编译对象为 `QrV2_566c...o`（136904 bytes，SHA-256 `8c7ecfb6...13723e`），配套 JSON SHA-256 `b87e4d98...ab500`。这证明失败张量、源码和实际安装包属于同一套 MX QR 交付物；仍需独立修复 kernel A/B 才能完成严格因果闭环。

### 用户新增修复门禁
- loss：与指定 GPU log 逐 step 计算相对偏差，所有 step 必须 ≤2%；任何一步超限即拒绝。
- 耗时：用户要求“不能偏差太多”，但尚未给出比较对象与数值阈值。在阈值明确前可继续根因诊断和机制验证，不得对修复作最终 PASS。
- 暂定建议（待用户确认）：相对当前 NPU 合格基线总耗时回归不超过 2%，同时继续报告相对 GPU 的 Iter2–30 总耗时、普通步和 QR 周期步。

### 当前二分假设
1. **算子合同故障**：某些训练上下文中 MX 返回 nonfinite、非正交、非三角或错误重构结果；可能与 current device、stream、workspace、调用顺序或末 tile 生命周期有关。
2. **集成合同故障**：MX 返回数学上有效的 QR，但与 torch/历史 Q 的基选择不同；SOAP 对 raw Q、排序与跨周期状态连续性敏感，导致后续状态或 loss 分叉。此时不能把“大逐元素差异”直接归因于算子错误。

## Requirements
- 8 卡昇腾 NPU，容器仅 `mapqr-leicheng`。
- 双门禁：耗时相对 CPU FP64 SOAP 大幅下降，且逐步 `loss` vs GPU `|Δ| ≤ 2%`。
- 吞吐目标：同合同 8 NPU : 8 GPU 达到 1:1 或更好。
- 不改远端已有驱动/CANN/PyTorch/torch_npu 版本。
- 远端产物不拉本地；分析原位完成。

## 2026-08-19 GitHub 发布审计
- Git 仓库根目录是 `C:\project\win-project-backup`，当前项目 `DongFeng` 是其子目录；兄弟目录不属于本次发布范围。
- 当前分支 `main` 跟踪 `origin/main`，远端是现有 GitHub 仓库。
- `机器IP.md` 含项目规则禁止在提交中重复暴露的连接信息，必须保留在本地且不得纳入本次提交。
- 本地 GBrain 项目标记是客户端绑定状态，不属于项目源文件，不得纳入提交。
- 未跟踪内容包含缓存、依赖副本、原始 trace、训练日志和张量样本；需先按安全与 GitHub 文件限制形成排除清单。
- GitHub CLI 已安装为 2.97.0，但尚未登录 GitHub；远端写入前需要可用认证。

## Current Blocking Finding: STEP-265 两个独立根因

### 问题 A：精度从 28/30 掉到约 11/30（不是社区 QR 独有）

| 实验 | QR | 跨 rank | PASS≤2% | step30 vs GPU |
|---|---|---|---|---|
| STEP-245 | HEAD 610 行 SOAP + DIST_QR=1 | broadcast | 28/30 ≤1%，30/30 ≤2% | +1.32% |
| STEP-254 | 63861df 风格 CPU FP64，无 broadcast | 各 rank 独立 | ~11/30 | 中后期偏高 |
| STEP-256 | mx QR + DIST_QR=1 | broadcast | 11/30，无 NaN | +25.38% |
| STEP-258 | CPU FP64 QR + DIST_QR=1 | broadcast | 11/30，无 NaN | +25.24% |
| STEP-274 | mx QR + 192 bypass，无 broadcast | 各 rank 独立 | **30/30 ≤2%**，23/30 ≤1% | **+0.72%** |

STEP-256 与 STEP-258 轨迹几乎重合（当时坏 QR + broadcast）。**修好 192 QR 后（STEP-274 bypass、无 broadcast）当前工作树即可 30/30 ≤2%。** 此前把 11/30 整段算到 SOAP 工作树上，部分是被坏 QrV2 带偏。

### 问题 B：无 broadcast 时后段 NaN（已定位到算子）

STEP-260 在每次 `mx_driving_cloud.linalg.qr` 后检查有限性与 `Q@R≈A`：
- 第一次周期（opt_step=10）4408 次调用
- 4400 次 OK：A/Q/R 全有限，`max|Q@R−A| = 7.2e-6`
- 8 次 BAD：shape 全是 `[192,192]`；A 有限，Q/R 非有限；ranks 0–7 各 1 次
- 同周期 256 次 192×192 里 248 次正常；5120/2560 大阵这次没有非有限

因果链：坏 Q 写进 SOAP 状态 → 后续投影飞掉 → 约 step16 起 `loss: nan`。开 `SOAP_DIST_QR=1` 后 NaN 消失，是因为坏结果被 rank0 正常 Q 覆盖，不是算子修好了。

STEP-266 对照 CPU FP64 SOAP 后，进一步排除 SOAP I/O：
- A 全有限、无 0/denormal，`absmax≈7.91e-8`，`cond2≈1763`，8 rank 哈希相同
- Q 非有限正好落在列 128–191（192×64）；R 同 64 列
- 同 A 的 numpy CPU FP64/FP32 QR 均成功
- STEP-257：CPU 预处理 + mx QR 仍 NaN；STEP-258：CPU FP64 QR 无 NaN
- 结论：不是 SOAP 把输入喂坏或把输出用坏，是 mx QR 在 192×192 最后一个 64 列 tile 生成非有限 Q/R；小幅值是触发条件

复现物：本地 `step260_qr_bad_tensors/`，已传到同事机 `/home/ubuntu/` 的 `rank{0-7}_step10_ind0_192x192_BAD.pt`。

## STEP-268：192×192 加严复现（后 8 卡，mapqr-leicheng）

同一份 STEP-260 BAD A + 同 shape SAMPLE + 幅值扫描 + 邻域尺寸，共 53 次独立进程调用 `mx_driving_cloud.linalg.qr`。

- **47/53 通过**：含 SAMPLE 192（`absmax≈0.49`，recon `6.4e-8`）、identity、良态 QR、随机 192、BAD 的 1e-4～1e8 缩放、以及 64/128/160/191/192/193/224/256。CPU FP32 全程有限。
- **同一份 BAD A 不稳定**：npu0/npu1 冷跑可以算出有限 Q/R（recon `2.0e-14`）；npu2–7 同一输入 AICore 崩溃 `507015`。首轮 in-process 在物理 device 8 上直接崩：`QrV2_*_mix_aic`，`MTE instruction DDR address out of range`。
- 训练里看到的「最后 64 列 NaN」和这次「有时算出、有时 kernel 崩」是同一算子的两种失败态，不是 SOAP 输入非法。
- 结论给同事：请用 `rank0_step10_ind0_192x192_BAD.pt` 的 A 在 192×192 上反复跑 `mx_driving_cloud.linalg.qr`，重点查 64-tile 最后一块 panel 的 MTE 越界。

## STEP-269：QrV2 源码与最后一块 tile（进行中）

安装包路径：`mx_driving_cloud.linalg` 实际是 `ops/linalg.py`；kernel 是 MIX `QrV2`（`qr_v2.cpp`）。

Python 包装：
- `QR_AICPU_THRESHOLD_SHAPE = 80`：任一维 ≤80 走 `torch.linalg.qr`，所以 64×64 根本不进 QrV2。
- `BLOCK_TILING = 64`：`lda = max(m,n)` 再 pad 到 64 倍数。**192 已对齐，pad=0**，kernel 看到的就是 192×192、`blockp=3`。
- 然后 `mx_driving_cloud._C.qr`，Q/R 再切回原 shape。

`Process()` 每个 k：
1. `GEQRT(k)`
2. `InitTaskTiling(k + 1)` 再 `LARFB`
3. 仅 `coreId==0` 再 `CalcQForLARFB` 并把 `qLocal` 拷到 `colQGm`
4. `i=k+1..blockp-1` 做 TSQRT/SSRFB；最后一块 tile 这个循环为空
5. `CalcCurrentQ(k)` 写当前列块

最后一块 `k = blockp-1`（192 时 k=2，列 128–191）：
- `InitTaskTiling(3)` 得到 `colNum = blockp-k = 0`，`useCoreNum=0`
- `LARFB` 对**所有核**走 `coreId >= useCoreNum` 早退：把 `a/t/v` 从 TQue `DeQue` 后 `FreeTensor`
- 随后 **core0 仍调用 `CalcQForLARFB`，内部 `DataCopy(..., vLocal, ...)`**。`vLocal` 刚被释放。这与训练里「只有最后 64 列非有限」和 STEP-268 的 `QrV2_*_mix_aic` / `MTE instruction DDR address out of range` 对得上。
- `InitTaskTiling` 在 `colNum <= cores` 分支不给 `tailRepeatNum` 赋值；`formerNum=0` 时所有核走 else，读未初始化 repeat。最后一块恰好命中。

STEP-269 前 8 个布局用例（npu0、warmup=0、独立进程）全部 `ok=True`，含非连续 `t_only`。warmup 0–128 亦全过。**8 卡 replay**：npu0/1 有限 recon `1.95e-14`；**npu2–7 全部 507015**（与 STEP-268 一致）。layout/warmup **不是**触发条件。

## STEP-270：设备分域 — QrV2 在后 6 张卡上整类失效（2026-08-18）

`ASCEND_RT_VISIBLE_DEVICES=8–15` 下 8 卡均为 `Ascend910_9362`；基础 `torch.ones` sync 8/8 正常。

| 用例 | npu0 | npu2 |
|---|---|---|
| identity/randn/sample 192 | OK | **507015** |
| BAD 192 | OK | **507015** |
| BAD pad→256 | OK | **507015** |
| 128×128 / 191 / 193 | OK | **507015** |
| **last64（64×64）** | OK | **OK** |

`linalg.py` 规定 `min(m,n)≤80` 走 `torch.linalg.qr`（AICPU），**只有 >80 才进 QrV2**。npu2 上唯一通过的是 64×64，说明崩溃来自 **QrV2 在 visible npu2–7（物理 device 10–15）上的 MIX AICore 路径整体不可用**，与 BAD 数值、layout、warmup 无关。

8 卡 BAD replay：npu0/1 OK，npu2–7 全崩（15/29 失败均为此类）。

## 根因结论（可交给算子同事）

**双根因，均已钉到具体条件：**

1. **设备分域（主因，解释 507015 / rank2–7）**  
   自定义 `QrV2` 在 visible **npu2–7** 上对任意 `max(m,n)>80` 的矩阵（含 identity、随机、SAMPLE、BAD、pad256）同步即 **507015**；≤80 的 AICPU 回退正常。8 卡 SOAP 训练每 rank 独占一卡，rank2–7 必踩此路径。

2. **最后一块 64-tile 算法缺陷（次因，解释 npu0–1 训练 NaN 末 64 列）**  
   `qr_v2.cpp` 在 `k=blockp-1`（192 的 k=2）时 `InitTaskTiling(k+1)` 得 `colNum=0`，LARFB 全核释放 `vLocal` 后 **core0 仍 `CalcQForLARFB`**；与 STEP-260 列 128–191 非有限一致。冷启动独立进程在 npu0 常成功，训练 in-process 高频 QR 更易触发 NaN。

**已排除：** SOAP I/O、stride/view、storage offset、warmup 0–128、BAD 专属数值（npu0 上缩放/pad 均 OK）。

**给同事的复现：** 容器内 `ASCEND_RT_VISIBLE_DEVICES=10`（或 npu2），`mx_driving_cloud.linalg.qr(torch.eye(192).npu())` 即崩；对照 npu0 同调用成功。BAD `.pt` 仅作训练态样本，非崩溃必要条件。

## STEP-272：前 8 卡 eye(192) — 换卡不能规避

前 8 卡当时空闲。`ASCEND_RT_VISIBLE_DEVICES=0–7`，**关闭** STEP-271 bypass，独立进程 `eye(192)`：

| visible npu | 物理 device | 结果 |
|---|---|---|
| 0 | 0 | OK，recon=0 |
| 1 | 1 | 有限但 **recon_max=1.0**（静默算错） |
| 2–7 | 2–7 | **507015** |

与后 8 卡（phy 8 OK / 9 OK / 10–15 崩）对照后，应修正 STEP-270 的「phy 10–15 特有」说法：

**QrV2 失败跟 visible 组内的 npu 下标有关（npu2–7 必崩），不是某几张物理卡坏了。** 换前 8 卡不能当正式方案。

## STEP-274：bypass 30 step vs GPU（后 8 卡，无 broadcast）

`MX_QR_VALIDATION_BYPASS=1`，192×192 走 `torch.linalg.qr`，工作树 SOAP 无 `SOAP_DIST_QR`。

| 门禁 | 结果 |
|---|---|
| 完成 | rc=0，30/30，无 `loss: nan` |
| ≤1% | **23/30**（失败：13/14/16/18/19/24/29，最差 step24 +1.55%） |
| ≤2% | **30/30** |
| step30 | NPU 225.5574 / GPU 223.9486，**+0.72%** |
| Iter2–30 | NPU **372.9 s** / GPU **139.6 s** = **2.67×** |
| CPU FP64 对照 | STEP-238/246 为 865–891 s（6.2–6.4×）；本跑约减半 |

相对此前无 bypass 的 STEP-259/260（后期 NaN）与 mx+broadcast 的 11/30（step30 +25%）：**只绕开坏 QrV2、不开 broadcast，当前工作树即可 30/30 ≤2%。** 吞吐未到 1:1，Iter4 SOAP 初基 208 s 占 Iter2–30 的 56%。

## Dual Gate Rescore (2026-08-18)
- 仅 STEP-238 / STEP-246（63861df CPU FP64 双轴）与 **STEP-274（当前工作树 + 192 QR bypass）** 达到 **30/30 ≤2%**。
- CPU FP64：Iter2–30 约 865–891 s，相对 GPU 140 s 为 **6.2–6.4×**。
- STEP-274：Iter2–30 **372.9 s，2.67×GPU**；精度最差 +1.55%。
- 快路径 HEAD+one-sided=1024 最接近 GPU 耗时（220 s，1.58×）但当时 16/30、最差 +11.7%。
- 不能靠 `fb979b2` 亲和栈同时满足双门禁。精度合同冻结为 63861df SOAP 数值；STEP-274 说明 **修好 192 QR 后，当前工作树也能过 2%**。

## SOAP Affinity Commit Attribution
- 根因提交：`fb979b28 【npu性能优化】SOAP预条件器NPU亲和优化`，父提交 `63861df`。
- 三条语义变更打成一个 commit：
  1. 正式 config `one_sided_dim_threshold=1024`
  2. 初基：CPU FP64 `eigh` → 设备单位阵 + 立刻 QR
  3. 周期更新：CPU FP64 mm/QR/`argsort` → 设备 FP32 + `stable=True`
- SOAP 窗 `pts` 拉开的最小变量是 one-sided=1024。
- 逐步 total loss 中后期偏低：关 one-sided 后仍在；整文件回到 `63861df` 才接近 GPU。
- 已排除：`SOAP_STALE_Q_K`、foreach/pin/GeometricLoss、仅 MSDA、仅 QR kernel。
- 用户禁止 overlay/`git restore` 抹掉 `fb979b2`；修精度须新 commit。
- 工作树已被 STEP-259/260 诊断改写，不能当正式基线。权威 HEAD 上次核验为 `669a138`。

## STEP-280：24 个 SOAP shape 逐个 vs CPU FP64（2026-08-19）

88 例独立进程，金标 `torch.linalg.qr(A.double().cpu())`，>80 强制 QrV2（无 192 bypass）。

| 路径 | 设备 | 结果 |
|---|---|---|
| ≤80 AICPU | npu0，eye/randn/small | 全 PASS |
| >80 QrV2 | npu0，eye/randn/small + BAD192/SAMPLE192 | **全 PASS**（合成/冷跑） |
| >80 QrV2 | visible **npu2**，eye | **14/14 FAIL 507015**：96,120,128,160,192,220,256,352,440,512,768,1024,2560,5120 |

结论：问题不是「某几个 shape 的合成输入算错」，而是 **凡进入 QrV2 的 shape 在 visible npu2–7 上都崩**。训练态末 tile NaN 仍是数据/进程态依赖，孤立冷跑连 BAD192 也可能有限。

## STEP-299：同一 BAD tensor 的七张物理卡单独可见测试（2026-08-19）

- 环境：42 远端、完整名称 `mapqr-leicheng`；官方 `mx_driving_cloud.linalg.qr`，无 192 bypass。
- 输入：本地 `rank0_step10_ind0_192x192_BAD.pt` 上传副本，SHA-256 与本地一致；A `[192,192]`、FP32、全有限、`absmax≈7.91e-8`。
- 模式：后 7 张物理卡逐一作为唯一 `ASCEND_RT_VISIBLE_DEVICES`，每个独立进程内 `device_count=1`、显式 `torch.npu.set_device(0)`，调用一次 QR。
- 结果：7/7 的 Q/R 全有限；Q/R NaN、+Inf、-Inf 均为 0；无 crash、无 507015。每卡 `recon_absmax≈1.954e-14`、相对 Frobenius 残差 `≈2.458e-7`、正交误差 `≈5.96e-7`、R 下三角最大绝对值 0。
- 结论：该 BAD tensor **不会在每张物理卡的真正单卡可见模式下必现 NaN**。结合 STEP-285 的 8 卡同时可见结果，故障更像与多设备可见时的 logical device / current-device / workspace/stream 上下文有关，而不是某几张物理卡本身损坏。训练态高频调用导致末 tile NaN 仍未被本次一次冷调用排除。

## STEP-301：纠正 507015 的 current-device 误归因（2026-08-19）

- 对照模式：`ASCEND_RT_VISIBLE_DEVICES=8–15` 保持 8 卡同时可见；logical npu1–7 各自独立进程，在搬运 A 与调用 QR 前显式执行 `torch.npu.set_device(k)`，并断言 `current_device==k`。
- 结果：7/7 的 Q/R 全有限，NaN/+Inf/-Inf 均为 0；crash=0、507015=0。七卡 `recon_absmax≈1.954e-14`、相对 Frobenius 残差 `≈2.458e-7`、正交误差 `≈5.96e-7`、R 下三角误差 0；输入哈希与 STEP-299/本地一致。
- 旧 STEP-285 harness 的关键缺口：使用 `A_cpu.to(f"npu:{npu}")`，但没有先 `torch.npu.set_device(npu)`。自定义 QrV2 依赖 current-device/stream/workspace 上下文，输入设备与 current device 不一致时可触发 MTE 地址越界和 507015。
- **纠正结论**：此前“visible npu2–7 上 QrV2 整类失效”不是算子对 logical device 的固有设备分域缺陷，而是诊断 harness 的设备上下文未绑定。物理卡 9–15 单卡可见与 8 卡可见正确绑定两种模式均正常。
- 仍未推翻：训练内曾真实 dump 到 Q/R 末 64 列非有限，且 `qr_v2.cpp` 最后空 LARFB 路径存在释放后继续使用的源码风险。一次冷调用 7/7 通过只能排除设备分域结论，不能排除训练态高频/状态相关 NaN。

## STEP-303：正确 `set_device` 下的高频 BAD192 重放仍未复现 NaN（2026-08-19）
- 环境：42 远端、完整名称 `mapqr-leicheng`、`ASCEND_RT_VISIBLE_DEVICES=8–15`，官方 `mx_driving_cloud.linalg.qr`，无 bypass。
- 模式：同一 `rank0_step10_ind0_192x192_BAD.pt`，在 logical `npu0/npu1/npu2` 上分别显式 `torch.npu.set_device(k)`，单进程内连续调用 512 次 QR；逐次检查 Q/R 有限性、507015 和重构误差。
- 结果：3/3 跑满，Q/R 全有限，无 NaN/Inf，无 507015；三卡 `recon_absmax` 一致约 `1.954e-14`。
- 结论：在“8 卡同时可见 + current device 正确绑定 + 高频重复调用”这个更接近训练态的受控子场景里，历史 BAD192 仍**不能**单靠 QR 重放复现 NaN。现阶段更像是训练主链路中的额外上下文因素参与了触发，例如 QR 前后的 stream/context、张量生命周期、调用位置或与其它算子交错的状态。

## STEP-334：QR Call Stack（2026-08-20）

- `operator_details` 有 Call Stack、**无 Step Id**（窗口=iter10–16）。
- LinalgQr 仅两条 Python 路径：`_qr_finish←_stale_q_submit`（多数）与 `get_orthogonal_matrix_QR` 同步 fallback（少数但 Device Self ~137s）；**无 install 栈**。
- 与 332/333 一致：显式 install sync 不是触发点；iter14 kernel QR 是侧流工作在时间轴上的兑现。

## STEP-332：install query 实测（2026-08-20）

- rank0：step 4/14/24 全部 `query_true`，`sync_ms=0`；iter14/24 仍 ~143s。
- per-factor event 优化无效；331 的「install synchronize 阻塞」推断被 332 修正。
- 实验资产清单见 `操作步骤.md` STEP-333 归档节。

## STEP-331 / STEP-330

- 331：iter14 QR kernel 159.6s（profile）；原始 profile 已删，摘要保留在 `操作步骤.md`。
- 330：k=0/k=4 同相位 iter4/14/24 长尾。

## Standing Performance Facts
- 永久基线：`63861df 【loss对齐】随机性移除`。
- 固定环境内多数单一严格等价边界已筛完或拒绝；1:1 吞吐仍未达到。
- SOAP CPU QR / host 空洞历史上是第一性能瓶颈；社区 QR 目前被 NaN/精度合同挡住，不能直接当性能解。

## STEP-330：SOAP_STALE_Q_K=0 vs 4 A/B（2026-08-20）

k=0 与 k=4 均在后 8 卡正确运行（`ASCEND_RT_VISIBLE_DEVICES=8..15`）。

| iter | k=0 | k=4 |
|------|-----|-----|
| 4 | 166.7s | 162.9s |
| 10 | 4.4s | 4.5s |
| **14** | **163.7s** | **140.8s** |
| 20 | 4.4s | 4.4s |
| **24** | **163.1s** | **140.5s** |
| 总计 | 758.7s | 625.2s |

**关键结论**：k=0 同样在 iter14/24 爆炸（163s），与 k=4 同相位。"仅 stale-Q install 导致 14/24"**被证伪**——但长尾仍是 QR 相关（见 STEP-331）。

## STEP-331：iter10 vs iter14 原位 profile 根因定位（2026-08-20）

后 8 卡、`SOAP_STALE_Q_K=4`、rank0 wait8/warmup1/active7，kernel_details.csv 原位分析：

| 训练 iter | 墙时（profiler step_id） | `aclnnLinalgQr_QrAiCPU_Qr` | 占 kernel 比 |
|-----------|--------------------------|----------------------------|-------------|
| **10** | ~1.79s（step_id 9） | **0 ms**（无 QR kernel） | ~0% |
| 11–13 | ~1.86s | 0 ms | ~0% |
| **14** | **161.5s**（step_id 13） | **379 次，159.6s** | **98.8%** |
| 15 | ~3.5s（残留） | 172 次，1.6s（install 尾迹） | 46% |

**根因已用 trace 钉死（k=4 生产路径）**：
- Iter10 `_stale_q_submit` → 侧流 enqueue QR，**默认流几乎无 QR 墙时**（0 ms）
- Iter11–13 侧流后台跑 QR，主路径正常
- Iter14 `_stale_q_install_if_due` → `event.synchronize()` → 379 次 `aclnnLinalgQr_QrAiCPU_Qr` **一次性串行兑现 ≈160s**

STEP-330 k=0 的 iter14/24 长尾仍是 QR，但原因不同（无 stale-Q 时走同步路径 `get_orthogonal_matrix_QR`）。

## Technical Decisions
| Decision | Rationale |
|---|---|
| 远端连接强制锁定 42 机器 | 用户明确要求防止因其他机器访问而串机；公共 helper 同时校验配置末段与连接后主机身份 |
| 精度合同 = 63861df CPU FP64 双轴 SOAP | 唯一 30/30 ≤2% 路径 |
| 不把 broadcast 当原始 SOAP | 只掩盖 NaN，用户已拒绝 |
| 不 overlay `fb979b2` | 正式修复必须新 commit |
| 社区 QR NaN 交给算子侧 | 有限 A 出非有限 Q/R，不是模型用坏 |
| STEP-283 清仓库内测试残留 | 只删 git 仓库内未跟踪 diagnostics/kernel_meta/trace；共享盘 diagnostics 与训练产物不动 |
| STEP-284 只提交 soap.py | HEAD `9565044`；相对 `669a138` 仅官方 QR 替换；未 push |
| STEP-285 同事单测「没问题」 | npu0 冷跑 8/8 有限 recon≈2e-14；同 8 份 A 在 visible npu2–7 48/48 崩 507015；npu1 7/8 有限但 recon≈A.absmax |

## Issues Encountered
| Issue | Resolution |
|---|---|
| 同事机账号写成 `ubantu` | 实际是 `ubuntu` / `/home/ubuntu` |
| 跳板机到同事公网机超时 | 本机下载后再直传 |
| 规划文件过大，hook/恢复成本高 | 2026-08-18 备份后重建精简版 |
| 新 NPU 机无 `mapqr-leicheng`，且目标卡被占用 | STEP-294 停在前检；禁止改用宿主机或其他容器，等待正确机器/环境与空闲卡 |

## Resources
- 备份：`planning_backup/2026-08-18/{task_plan,findings,progress}.md`
- BAD tensors：`step260_qr_bad_tensors/`
- 修正后的算子复现包：`qr_operator_repro_for_colleague_step301_corrected.zip`（旧 `qr_operator_repro_for_colleague.zip` 的 507015 复现口径已过时）
- 操作记录：`操作步骤.md`（STEP-265 及以前）
- 远程连接：只读 `机器IP.md`，不把凭据写入本文件
## STEP-338：最新提交与首个 loss 越界点只读审计（2026-08-21）

- 远端 `ascend_npu_optimize` 当前 HEAD 为 `3a1d763`；相对直接父提交只修改 `soap.py`，删除 MX import，并把两处 MX QR 调用回退为 `torch.linalg.qr`。
- 当前远端工作树并非纯 HEAD：`soap.py` 仍叠加未提交的 stale-Q/per-factor Event 异步调度与诊断改动，但 QR backend 仍为 torch。当前训练结果不能只归因于 `3a1d763`。
- 最新带 `tested_commit` 的正式 30-step MX QR 记录为 STEP-359（tested commit `230d4f5…`）。按 `abs(NPU-GPU)/abs(GPU) <= 2%` 逐 Iter 对齐，step 11 为 `-1.3405%`，step 12 首次失败为 `-2.3145%`；全程 11/30 通过。
- STEP-362（192 走 torch）、STEP-363b（调度实验）以及旧 STEP-326 torch 基线也均在 step 12 首次失败。由现有日志不能把 step 12 总 loss 偏离归因于 MX QR；下一步需拆解子 loss、样本顺序与 step10～12 SOAP 状态。
- STEP-359 Iter2～30 总耗时相对 GPU 为 `+17.47%`；剔除显著 QR 慢周期仍为 `+7.25%`。该实现含逐次 synchronize 与 `.item()`，只能说明当前候选不满足耗时门禁，不能代表无同步修复内核的最终性能。
- step12 的总 loss 差异中，`map_pts + map_pts_normal` 占约 `90.92%`；STEP-326 torch QR 在同一步呈同型分量偏移，进一步反对“step12 首败由 MX QR 单独导致”。
- 时序修正：对 iter1 起有梯度的 state，QR cadence 在 iter1/11/21 的 optimizer 内发生；iter11 末尾生成的新 Q 在 iter12 optimizer 才首次消费，最早影响 iter13 forward。故 iter11 QR 不能导致已经完成的 iter12 forward loss。
- 严格 loss 因果测试应先做 device-only 摘要 A/B，并加入 torch-vs-torch control；以 QR contract、特征值簇子空间、project_back、参数增量和后续 forward 定义语义分叉，不能比较 raw Q。首分叉后才进行完整 tensor dump。

## STEP-372：v5 Matmul 位置合同根因候选（2026-08-21，历史判断）

本节是 STEP374 之前的候选判断；其“未验证/下一步”已被下节 runtime timeout/trap 结果覆盖。

- **已确认事实**：v4 真机 `R nonfinite=16448`，精确等于 192×192 上三角元素数减首个 64×64 R00 上三角元素数；污染从首次 LARFB 路径开始的解释与计数一致。
- **已确认源码错误**：`vtvMatmulObj` 的 A 声明为 `VECIN`，v4 CalcQ 第二乘却传 GM；`qaMatmulObj` 声明 `VECIN/VECIN/GM`，LARFB/SSRFB 现有实参均为 `VECIN/GM/VECIN`。
- **合理推测**：这两个错配是当前能同时解释首污染位置和非有限计数的最强根因候选。
- **未验证假设**：将两者修正后一定能消除现场 NaN。该假设只能由目标 CANN OPC 编译、v5 concrete AIC 命中和同一真实192输入的 finite+重构/正交门禁证明或证伪。
- **当时的延后候选**：GEQRT ReduceSum scratch alias 和 V→S 同步风险保留为 P1，但不混入 v5。当时计划 finite 失败后做 T/V/q/a probe；STEP374 实际在Q/R产生前runtime失败，当前改为先做PC/源码映射和单观察边界probe设计。

## STEP-374：v5 首次真实调用发生 AICore timeout/trap（2026-08-21）

- **已确认事实**：8/8 rank 的 world8、物理设备8–15、shadow/OPP、固定192输入前置门禁通过；8/8 设备错误均点名 `QrV2_matmul_position_fix_v5_0_mix_aic`。
- **运行现象**：第一次调用在 `torch.npu.synchronize()` 阶段统一触发错误码507014、AICore timeout/trap和MTE错误；ready=8、done=0、traceback failure=8，Q/R未转回CPU。
- **不能推出**：没有 finite、重构、正交或 profiler identity 结果，不能称为“v5仍产生NaN”，也不能仅凭MTE错误唯一归因到某条AscendC语句。
- **当前根因层级**：问题已收敛到 v5 kernel内部执行/同步/MTE访问或流水活性；`vtv` 直接消费 `vLocal`、`qa` 的 `VECIN/GM/VECIN` 类型变更、既有同步/所有权缺陷及其交互都必须分别隔离，当前证据不能排除其中任何一类。
- **清场证据**：postflight/finally PASS，端口34359、后8卡和9个受管PID/starttime均清零。禁止自动复跑；先做源码/PC映射和单变量诊断设计审核。单变量probe每次只能有一个观察边界或一个可逆delta，必须锁定独立identity/SHA及workspace/tiling/event增量，不能同时导出T/V/q/a后仍称单变量。

## STEP-375：PC映射与delta1-only诊断候选（2026-08-22）

- **PC强证据**：v5 ELF的AIC symbol为`0..0xe134`，AIV从`0xe134`开始；运行时AIC/AIV `pc_start`差值也精确为`0xe134`。AIC相对PC形成两簇：rank0/1/4/5为`+0x137c`、rank6为`+0x1368`，rank2/3为`+0xb838/+0xb848`；AIV异常的rank1/2/6/7均为函数内`+0x8a4c`。
- **映射边界**：对象无DWARF/`.debug_line`，仅有kernel级AIC/AIV符号，现有目录无map/asm/line-info；因此只能高置信映射到函数和offset，不能映射到`CalcQForLARFB`或`UpdateAForLARFB`源码行。MTE字段12次均非零但数值不同，不能当成同一故障地址。
- **delta1审计**：`vLocal`容量、对齐、队列生命周期、core0作用域及释放顺序未发现确定性错误；SSRFB已有直接LocalTensor作为第二乘A的相同结构。delta1仍可能触发流水等待，但当前证据弱，不能由generic MTE错误直接归因。
- **delta2审计**：`qa=VECIN/GM/VECIN`同时改变B的GM读取和C的Local写回lowering。LARFB/SSRFB的GM-B前已有MTE3→MTE2依赖，但现有源码无法证明`IterateAll(aLocal)`后的CUBE/FIX→UB→MTE3完成语义；delta2是活跃嫌疑，不是已证明根因。
- **正交矩阵**：v4=`2213dbae…4614b`，delta1-only=`ef5db14e…ce180`，delta2-only=`e352ac31…3b003`，v5=`e6ccbb84…bc3b7`，四个SHA唯一。
- **最小下一步**：先实现delta1-only（相对v5只撤回delta2）的独立诊断identity，用来回答qa lowering是否是trap必要条件。即使trap消失，也只说明delta2或其交互触发执行故障，不证明legacy qa语义正确或NaN已修复。
- **STEP376首次执行根因**：唯一远端尝试在目录检查脚本的quoted path与`]`间缺空格，shell把`<path>]`视为单一token，精确触发rc=2与`[: missing ']'`。现场无诊断目录、无上传、无OPC进程，installed/runtime未触达；这不是CANN或算子编译失败。
- **首次失败修复态（历史）**：controller修复后曾回退`BUILD_READY=False`，46/46本地测试通过，并要求重新phase-transition；该状态已由下一条retry2武装状态取代。adapter始终毒化`package/all`并禁止package/wheel/install/modify-installed/NPU/train。STEP376仍只是判断delta2必要性的诊断链，不是NaN修复。
- **retry2当前态**：新phase-transition已允许以全新DIAG_NAME武装，当前`BUILD_READY=True`、46/46 PASS，candidate与输入SHA未变；attempt2尚未执行，等待armed-state终审。即使构建成功也只得到`diagnostic_built_unvalidated`，不能升格为修复。
- **retry2执行结果**：唯一执行在首个before-snapshot拒绝官方OPC symlink alias；远端仅有6个上传输入，work/manifest/opc.log/object/kernel JSON/release输出均不存在，相关进程0，因此OPC未启动。最终regular target的SHA与container probe合同一致。不能用不同schema旧manifest声称installed/runtime逐项基线相等，只能按控制流确认本轮未进入任何修改路径。
- **最小修复方向**：不放宽base/adapter的non-symlink regular门禁；controller应从已验证container contract读取`opc.path`的resolved绝对路径，并统一传给pre/post snapshot和container build。这样alias后续改指不会换工具，target缺失或SHA变化仍fail-closed。
- **realpath修复结果**：上述最小修复已实现并独立审核P0/P1/P2=0。legacy OPC alias不再进入任何build/snapshot命令，锁定realpath经过特殊字符quote及alias换靶测试；当前`BUILD_READY=False`，尚无新的远端准入。
- **attempt3当前态**：新phase-transition已用全新DIAG_NAME武装，`BUILD_READY=True`；49/49加定向1/1通过，candidate/输入SHA/权限未变。尚未执行远端，等待armed-state终审。
- **attempt3结果**：唯一OPC构建成功并原位复算审计闭合。双SoC object/JSON字节一致、诊断identity和source/reverse/tool SHA正确、package永久禁止、release输出不存在、installed/runtime闭合、相关进程0。controller已重新关闭。该结果仅是`diagnostic_built_unvalidated`，尚无NPU runtime证据。
- **STEP377 shadow builder**：最终终审P0/P1/P2=0。完整锁定真实STEP376 manifest/原wheel/artifact，流式安全解包，完整树差分，dirfd/O_NOFOLLOW overlay，partial marker与manifest最后发布事务均闭合；只会生成`diagnostic_shadow_unvalidated`。尚未用真实attempt3产物远端执行，也未接NPU。
- **STEP377 worker**：薄适配终审P0/P1/P2=0。固定一次真实192路径，profiler task identity严格为diagnostic AIC=1且同名多hash拒绝，diagnostic gate regular/inode闭合，异常恢复主错优先；尚未接host/controller或运行NPU。
- **STEP377 host**：终审P0/P1/P2=0。8-rank diagnostic gate以token/inode和每rankack端到端闭合，ready/module/input/math/profile及失败finally门禁闭合，复用原STEP358 ownership清场。
- **remote controller当前缺口**：`NPU_READY=False`。真实连接/上传入口已接STEP357 helper，但6类容器动作仍为echo envelope，远端上传未复核，启动超时前无ownership，清场顺序及SFTP-open失败资源关闭未闭合；不得武装。

## STEP-340～341：loss gate 固化与 QrV2 修复候选编译（2026-08-21）

- 新增 `.codex-tools/step340_loss_gate.py`，修复旧 STEP326 comparator 的返回码反向缺陷。新 gate 对 missing/duplicate/nonfinite/零分母/任一步超过2%均非零退出，8个单元测试通过。
- 远端原位对权威 STEP359 日志验证新 gate：rc=1，11/30通过，step12首败2.3144829%，step28最大20.9940952%；与先前独立聚合一致。GPU JSON无子loss，因此本次 gate 的 sub-loss summary 不能作子项证据。
- QrV2 lifetime candidate 在隔离 diagnostics 中经官方 CANN `opc` 成功编译。候选源码 SHA=`5a4d140b…4105b`、object SHA=`b2ce04e9…18685`、JSON SHA=`b4e9fc49…92a06`；object/json唯一且非空，未见资源 overflow。
- 编译候选包含：tiling局部变量零初始化、LARFB local tensor 延后释放，以及当前 CANN const兼容。const兼容仅复制4个已证实为200B trivial-copy的 `TCubeTiling`，不复制未知布局的整个 `QrV2TilingData`，不使用 `const_cast`。
- 已安装 wrapper/source/object/config 的编译前后 SHA完全一致；候选未加载设备、未替换安装包、未训练。下一门禁是证明进程实际命中候选 object 后，做原版/候选的真实输入与训练上下文 A/B。

## STEP-351：shadow 冷测错误尝试与当前证据边界（2026-08-21）

### 已确认事实

- 首次编排在宿主机执行容器内 `site-packages/mx_driving_cloud` 路径的 `cp`，报 source 不存在；未启动 NPU/QR。容器内只读探针随后确认包路径实际存在，错误是执行命名空间用错，不是远端缺包。
- 本地预检命令曾因 AWK 花括号被 Python 格式化解释而 `KeyError`；错误发生在远端命令执行前。
- retry1 使用准备器不存在的旧参数名，`argparse` 在复制和 NPU 前退出。
- retry2 复用了硬编码 STEP338 candidate stem 的旧准备器，因此把存在的 STEP350 `.o/.json` 误判为 candidate pair missing；仍未复制 shadow、未启动 NPU。
- SHA 锁定的 STEP350 适配准备器加入后，retry3 的双 SoC overlay 准备通过；8 rank ready 和后8卡 live gate通过。release 后8 rank全部失败，`done=0`、`captures=0`、controller rc=122、launcher rc=1、postflight rc=0；安装态6个追踪文件运行后复核 PASS。
- retry3 只有通用 `QrV2` profiler证据，没有可验收的 `hash_dic + task_track` concrete AIC引用。因此尚不能证明设备实际执行了诊断内核。

### 合理推测，尚未验证

- 按 worker 控制流，首要怀疑点是 `decode(raw_R)` 的 magic/header/completion 门禁；也可能是诊断 concrete AIC 未命中或 `FlushDiagnosticTail()` 未执行。必须读取实际 traceback 后才能裁决。

### 明确不能得出的结论

- retry3 不是 Q/R 数学失败证据，也不是 T/V 生命周期证据。
- 当前没有抓到 Free 前/后的 T/V，不能声称已运行时证明 use-after-free。
- 不能用通用 `QrV2` profiler名替代具体 AIC身份；不能因为 shadow配置准备成功就推断候选二进制已执行。

### 方向校正

- 上下文抓取用于区分“输入A/QR公式问题”和“释放后状态被再次消费”，方向有意义；但借R下三角回传可能改变时序，必须先通过最小冷测的身份、header、载荷完整性门禁。
- 下一步不做shape扫描、高频重放或长训练。先原位读取retry3 traceback，并把A/Q/raw-R保存提前到`decode()`前；若下三角通道仍不可靠，改用独立诊断缓冲区。
## 2026-08-21：同条件 MX/CPU QR A/B 新合同

- 用户要求两层证据同时成立：同一训练条件/抓取时机的模型内 MX QR 与官方 CPU QR 双路径对照，以及同一现场输入 A 的 MX/CPU 单算子重放。
- 不能把“相同步骤编号”当作输入相同；必须以 rank、optimizer step、调用序号、输入 SHA256、shape、dtype、stride 共同闭合。第一次 QR 分叉后两条训练轨迹可能不同，因此模型内 A/B 必须从相同 checkpoint/RNG/batch 状态重启，或在同一次调用点从同一个 A 分叉。
- QR 的 Q/R 存在列符号不唯一性；除 finite、重构、正交、R 下三角和子空间合同外，逐元素 Q/R 比较必须先做符号对齐。
- npu1 与 npu2 共享工作目录。即使训练机器不同，直接修改活跃共享仓库仍会影响其他项目；必须在 npu2 容器内使用固定提交的唯一隔离副本/独立 worktree、独立端口和输出目录，只读引用数据与 checkpoint。
- 算子部门证据包必须保留原始失败 A/Q/R、CPU 官方 QR 同输入结果、逐文件 SHA256、异常坐标范围、环境指纹、可独立重放脚本与不可覆盖 manifest；全部远端原位保留，不下载到本地。
- 用户再次明确：主证据必须是训练时端到端路径中实际触发的算子 I/O，不能用训练后离线 replay 冒充。夹具必须证明 SOAP 真实调用点被命中；进入前保存 A，返回后立即保存 MX 或 CPU Q/R。单算子同输入 replay 只作为第二层复现证据。
- 保存时机是合同的一部分：每次目标调用完成后立即 `detach` 到 CPU，在唯一文件名下临时写入、fsync、原子发布并追加不可覆盖 manifest；发现非有限值后保留原始张量并停止覆盖。
- 远端只读查询若本地 exit0 但没有远端唯一成功标记、hostname/IP 和命令结果，必须判定为无证据；复杂 expect 命令需拆分，不能用空输出推断远端状态。
- npu2 容器只读核验：唯一精确容器 `mapqr-leicheng` 运行中；`/mnt/sfs_turbo` 以 RW 挂载，证明直接改共享仓库会跨机器可见。当前共享仓库分支 `ascend_npu_optimize`、HEAD `3a1d7633582d079a2f3e3ddba6fa2555c14da77f`，已有 11 个 tracked dirty 路径，SOAP SHA256=`dd3753f4291dcf523f1a92fb6e0a2247610714049924092707b04addaf430b94`，禁止本任务在该活跃工作树写入。
- 当前远端 SOAP 的未提交内容已出现 `SOAP_USE_MX_QR` 和“same-process dual QR”文字，可能是其他项目正在实现的诊断逻辑；在定向只读审计其语义前，不复用、不覆盖，也不把它当作本任务已完成证据。
- 用户明确本次训练使用“后8卡”，即逻辑设备/Phy-ID 8–15，对应 `ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`。`npu-smi` 的 NPU 4–7 每组含两个 chip，不能误表述成只有4张可用卡；此前结果实际显示后8逻辑设备均无运行进程。
- 旧 STEP-315 sitecustomize 仅保存 A，未保存实际 Q/R，并写入活跃共享 repo；不满足新合同。新实现改为隔离 SOAP 显式调用 `qrv2_training_capture.qr`，避免全局 monkeypatch，并以 input/started/output/complete 或 failed 独立不可覆盖文件及时落盘。
- STEP377 controller 文件闭包实现态19/19本地测试通过且`NPU_READY=False`，但独立审计为P0=2、P1=3，不能准入：route可用`..`穿越；shadow artifact/RECORD未精确绑定两个SoC包内路径与原wheel完整树；目录项集合缺读取后复核；本地hash后再读存在TOCTOU；installed扫描未覆盖额外SoC/目录/route。测试通过只证明已有断言，不证明安全合同完整。
- STEP377历次错误尝试均已保留为审核历史，不能用早期绿测覆盖。最终全链已关闭route穿越、artifact/wheel/config/RECORD/full-tree、目录竞态、ownership替换、PGID/PID复用、rank残留、gate先发布和cleanup短路问题。最终两路终审P0=0/P1=0；`NPU_READY=False`，结论仅是诊断控制链可进入phase-transition审核，不是QrV2已修复。

## 2026-08-22：attempt9定界与delta2-only实现

- attempt9的8/8 rank均有diagnostic AIC task reference=1、AIV=0；root是rank2在`evaluate_call -> torch.npu.synchronize`超时，其他ChildFailed/SIGTERM是级联。证据只能说已dispatch的device work未在观察窗返回，不能定位kernel内最后指令或证明NaN。
- delta1-only在不含delta2时仍超时，因而delta2不是attempt9超时的必要条件。下一有信息增益的单变量是撤销delta1并仅保留qa位置修正，不重跑已知能返回但nonfinite的字节相同v4。
- STEP384 delta2-only生成器已实现：只改qa声明，保留v4 per-core scratch/event，candidate/reverse SHA闭合，12/12 PASS；终审P0=0/P1=0/P2=1，可进入默认未武装的隔离构建wiring。

## 2026-08-22：STEP392 standalone 证据与端到端口径

- STEP392 原位只读聚合确认：8/8 rank每个唯一命中delta2 candidate AIC，AIV/旧identity为0；input/Q/R finite、重构、正交、R下三角和CPU QR投影均PASS。
- 但历史 STEP319–323 已经证明原MX QR也可standalone finite/重构PASS，而端到端MX仍在Iter6出NaN。因此STEP392只是独立路径安全门禁，不是训练修复证明。
- 最小正确方向：先做一次capture-off/profiler-off的低扰动真实30-step delta2 shadow训练，以原生loss/iter time判定；避免在每次QR后`.item()`同步而改变异步/stale-Q时序。
- 低扰动轮无法直接证明concrete kernel identity；若正确性通过，需再用首个真实训练QR的短profiler闭合identity，但该轮不用于性能裁决。
- STEP393 已锁 source commit/SOAP/STEP204/GPU oracle/loss gate；默认双闸门关闭。第一次聚焦测试失败为测试字符串预期陈旧，独立复核是 `TEST_EXPECTATION_STALE`，修正后同组1/1 PASS。
- 最终只读审查仍发现 pidfd/ownership 启动窗口 P0，以及 handoff 原子性、`TASK_QUEUE_ENABLE` 生产上下文保持、exact argv、rank↔NPU bijection、cleanup/loss完整校验等 P1。因此当前明确 NO-GO，尚未触网或运行训练。
- STEP393上述P0/P1最终已清零并获得仅针对夹具的GO。attempt1在训练/NPU前的archive shell条件因`path]`缺空格失败；独立复核确认控制器拼接错误，未触发算子。三个条件和生成命令校验已最小修复，旧目录不复用，等待attempt2增量审核。
- attempt2仍在调用远端archive前停止：新增校验器错误地按`;`拆包含quoted Python的完整script，造成`No closing quotation`。已改为纯条件helper局部自验；未训练/NPU，不自动attempt3。

## 2026-08-22：STEP393 attempt3 尚未触达训练及守卫根因

- attempt3 只完成 source/shadow 准备并拉起 Docker child；8 rank ready=0、start gate不存在、无rank ownership、无train log/loss/timing/result。因此它没有产生任何可评价 MX QrV2 的端到端证据。
- 独立原位审计确认当前 exact case/owned PGID/端口/后8卡进程均为0；但 cleanup postflight 缺失，故只记“当前现场清零”，不倒推协议清理PASS。
- 失败来自 STEP393 预筛先严格解析全部 `/proc`、后判断目标PGID。无关PID的合法 `pgrp=0/1` 会被STEP377严格身份规则拒绝并外冒，阻断目标组扫描。
- 最小修复仅放宽STEP393“非目标PID预筛”，目标PGID的starttime、身份变化、授权与发信号门禁仍fail-closed；3/3聚焦测试PASS。该结果只恢复E2E夹具可运行性，不是算子修复结果。

## 2026-08-22：STEP393 attempt4 摘要闭包遗漏

- attempt4在训练前pre-snapshot停止：backend嵌入旧guard SHA，而controller与实际guard已是新SHA；这是摘要传播/跨文件测试缺口，不是MX QR证据。
- 现场确认run目录及所有rank/train/loss结果均不存在，exact case/PGID/端口/后8卡均清零。
- 只同步摘要链并增加controller/backend/guard交叉不变量；独立Python审查P0/P1=0。attempt4不可复用，下一E2E必须使用全新目录。

## 2026-08-22：STEP393 attempt5 被漂移 config 门禁阻止

- attempt5启动了Docker launcher，但8 rank ready前退出；未执行torch_npu/shadow import/训练/MX QR，无loss或耗时证据，现场已协议清场。
- SOAP归档身份实际闭合；早先SOAP不匹配是审计NUL转义假阳性。
- 唯一真实静态失败是STEP193/204训练config：锁定`02aca0...`/145464 bytes，现场为`79c014...`/145465 bytes，且非单一尾换行差异。attempt5未修改该既有文件。
- 正确方向是找到已有canonical锁定副本，不是接受漂移文件或修改GPU oracle合同；未找到前禁止下一E2E。

## 2026-08-22：canonical恢复与attempt6边界

- 项目内基线可经两处唯一变换逐字节恢复canonical config `02aca0...`；新文件独立生成，未覆盖漂移合同。active absolute base又与锁定commit归档base做SHA等值闭包。
- attempt6静态/base/SOAP AST通过，但environment preflight为0字节；最小日志确认torch/torch_npu导入链ImportError，尚未执行Config.fromfile、launcher、rank、训练或MX QR。
- host waiter未及时消费bootstrap child rc1，表现为ready timeout 0/8。下一修正对象是结构化preflight failure发布及ready轮询同步检查bootstrap_result，不是QR实现。
