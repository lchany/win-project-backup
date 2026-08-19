# Findings & Decisions

历史全文已备份到 `planning_backup/2026-08-18/`。本文件只保留当前仍会影响决策的结论。

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

## Standing Performance Facts
- 永久基线：`63861df 【loss对齐】随机性移除`。
- 固定环境内多数单一严格等价边界已筛完或拒绝；1:1 吞吐仍未达到。
- SOAP CPU QR / host 空洞历史上是第一性能瓶颈；社区 QR 目前被 NaN/精度合同挡住，不能直接当性能解。

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
