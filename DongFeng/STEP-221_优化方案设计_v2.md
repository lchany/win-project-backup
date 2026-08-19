# STEP-221 优化方案设计 v2（2026-08-15）

## 背景与裁决输入
- STEP-220 唯一一次 world8 局部门禁已裁决 `REJECT_LOCAL_SCREEN`：Brockett+cubic polar 性能门（每周期净省约 22.29 s）与内存门通过，但真实投影作用 rel-L2 最坏 1.49、Rayleigh offdiag 超 0.3、2560 正交 6.13e-3 全部失败。结论：单步流形法无法在一个周期内追上 QR 的整周期基跳变，该候选关闭，禁止 eta/substeps sweep 重开。
- 严格位级等价轨道已穷尽（STEP-089~100、199、214/215）：同 shape batched QR 位级一致但更慢；Q-only geqrf/orgqr 非位级且触发 CPU fallback；Householder/TSQR/Gram-Schmidt/Cholesky-QR 重写均 `NO_GO_ALGORITHM_NOT_STATE_EQUIVALENT`；out-buffer、缩短 Step10 的 QR-对-QR 多流在正式 8-rank 回归；host CPU QR 是 GPU 时代旧路径（FP64 约 95.5 s），比 AICPU 更慢。
- 版本冻结：不升级 CANN/torch_npu；`aclblasSgeqrfBatched` 等新 primitive 仅登记为未来线索。

## P0-v2：SOAP 周期 QR 异步流水化（stale-Q，固定 k 步换入）
### 机制
- 周期步 t：按现行代码计算 QR 输入（power_iter 输出）后，将全部 543 次 `torch.linalg.qr` 提交到**独立 NPU stream**，不在当步等待；训练沿用旧 Q 立即继续。
- 固定延迟 k 步（所有 rank 相同、入合同），在步 t+k 边界同步 event 后，用与同步版**完全相同**的排序/后处理/状态代码原子换入新 Q。k 由 Stage A 实测 QR wall 确定（预计 3~5），硬上限 k ≤ frequency−1=9；event 未就绪则同步等待（受控 fallback）。k=0 即恢复原语义，作为回退开关。
- QR kernel、参数、stable sort、dtype、shape 白名单全部不变；STEP-096 已证明跨流 QR 位级一致。变化只有"新 Q 的生效时机"。

### 为什么这不是重开已关闭方向
- 已关闭的多流方案目标是缩短 Step10 自身（QR 与 QR 并行），受 8-rank AICPU 饱和与同步开销限制而回归。
- 本候选不缩短 QR，而是把 22.6 s AICPU 墙移出关键路径：普通步主要用 AI Core/host，AICPU 后台消化 QR，两者硬件资源大体正交。

### 已知风险（Stage A 必须量化）
1. 普通步自身的 AICPU 算子（如每步约 2048 次 AICPU ViewCopy）与后台 QR 争用 AICPU，可能拖慢前台。
2. 543 次 QR 的 host 下发（约 1~2 s）挤占前台 host（普通步本就 underfeed 75%）。
3. 语义变化：Q 生效延迟 k 步（相位平移）。DDP 下各 rank 状态相同、k 固定，rank 间一致性与可复现性可保证。

### 门禁链（预声明，任一失败即关闭；改 k 以外任何机制=新候选新审计；不得现场放宽阈值）
- **Stage A 微基准**（占卡约 10~20 分钟，非训练）：真实 23 类 shape 的 543 次 QR 后台流 + 前台混合负载（matmul+AICPU ViewCopy 型算子）并发实测，单卡与 8 卡并发两种。价值线：QR 可隐藏比例 ≥70% 且前台减速 <5%；实测得出 k。不达线直接关闭候选。
- **Stage B 静态包+独立审计**：仓库外 adapter/tool-root（沿用 STEP-220 已验证的 host-controller 布局），fail-closed、source contract、PGID-only 清理，与 STEP-216 同级。
- **Stage C 8 卡 30-step 单变量 A/B**：loss/grad 有限且曲线偏差在既有噪声带内；普通步不劣化（>5% 即失败）；周期步净省 >5 s（价值线，227 ms 仅为噪声下限）；checkpoint save/load/resume 等价（延迟窗内的 pending Q 必须在 save 前强制换入，保证 state_dict 语义不变）。
- **Stage D 876-step + 同 checkpoint 测试集**：吞吐较基线 0.58:1 提升到 ≥0.75:1；评测指标与基线等价；全程每周期记录换入延迟与 loss 漂移序列（审计缺口 2 的只读监控）。
- **提交**：Stage D 通过后单 commit `【npu性能优化】SOAP周期QR异步流水化（陈旧Q固定k步换入+同步回退）`。

### 预期上限（如实沟通）
- P0-v2 全成功摊销回收约 2.0~2.2 s/step，整体约 0.78~0.81:1；**单项到不了 1:1**。

## P1：普通步 host/underfeed（次序在 P0 后）
- 允许训练后，可采**一次** Level0 无栈、无 shapes 低扰动 profile（rank0、3 个普通步）做 launch-gap 唯一边界聚类；原位分析、摘要脱敏、raw 用后即删并复核为 0。
- 只有钉出"单一源码边界、功能不变、profiler-off >22.7 ms"才实施；NPUGraph/TQ2/pin 等已拒绝项不重开。
- 预期注记：GPU 侧 underfeed 也有 65.4%（NPU 75.3%），差值部分可能是宿主 CPU 架构差异，非项目代码可全额回收。

## MSDA：保持关闭
- 残差在固定 DrivingSDK 空间 FP32 主 kernel，项目可见上限约 6.2 ms < 22.7 ms 准入线；等厂商/隔离验收环境。

## 语义变更授权点
P0-v2 属于"训练语义时序变化"（Q 延迟 k 步生效），与 Brockett 同轨但数值安全性强得多（QR 数学完全不变）。按项目惯例，进入 Stage A 前需用户对该语义轨道明确批准。
【2026-08-15 已批准；Stage A 已通过：单卡+8 卡 hidden≥99.86%、前台减速≤0.14%、host 下发 0.019s、AICPU 每卡私有无跨 rank 争用；k=4。】

## Stage B 实现要点（基于权威 soap.py 只读分析定稿）

### 源码事实（`soap.py` 19,169 B / SHA `0e49429d...`，只读核验）
- `get_orthogonal_matrix_QR` 每因子做四件事：`est_eig=diag(oᵀ m o)` → `sort_idx=argsort(est_eig,descending,stable)` → **`exp_avg_sq=exp_avg_sq.index_select(ind,sort_idx)`** → `o=o.index_select(1,sort_idx)`；`power_iter=m@o`；`Q,_=linalg.qr(power_iter)`。函数末尾 `state['exp_avg_sq']=exp_avg_sq` 并返回新 Q 列表。
- 触发点在 `update_preconditioner` 末尾：`state['step']>0 and step%freq==0`（step 已在 `_step_foreach_chunk` 内自增），而 `update_preconditioner` 本身在 chunk 参数更新之后逐参数调用。
- 首次 `Q is None` 分支（identity basis + 一次 QR）另走 `step()` 内 `"Q" not in state` 路径。

### 确认的强制约束（与 v2 初稿相比的实质修正）
1. **exp_avg_sq 置换必须与新 Q 原子同装**。它不是"基变换重算"，而是由 `sort_idx` 决定的坐标重排。t 时刻置换、t+4 装 Q 会让 exp_avg_sq 坐标与在用的旧 Q 错位。正确做法：t 时刻只记录 `(ind, sort_idx)`，t+4 与新 Q 同一临界区内对**当时**的 exp_avg_sq 依序 `index_select` 重放。t~t+4 期间"旧 Q + 未置换 exp_avg_sq"自洽，t+4 后"新 Q + 已置换 exp_avg_sq"自洽。
2. **GG 原地更新与侧流读取存在竞态**。`GG.lerp_()` 每步在默认流原地改 `m`，若侧流跨步读 `m` 会数据竞争。因此切分点定为：`est_eig/sort_idx/o_permuted/power_iter` 全部在默认流同步算完（纯 matmul，AI Core，毫秒级），**只把 `linalg.qr(power_iter)` 提交侧流**。`power_iter` 是新张量、不与 GG 别名，竞态消除。
3. **pending 不重叠**：k=4 < freq=10，任一 state 同时最多一个 pending，须显式断言。
4. **首次 Q 初始化保持同步**（Q 必须先存在才能 project）。
5. 释放侧流用过的 `power_iter` 需 `record_stream`，避免 caching allocator 提前复用。

### 内存预算（须在 Stage B 实测）
543 因子 Σn² ≈ 71.4M 元素 ≈ 285 MB/份（FP32）。异步版稳态额外常驻 ≈ 新 Q 一份（285 MB）+ 少量在飞 `power_iter`；旧 Q 在同步版本已存在。需实测 allocated/reserved 增量，若超出既有 256 MB/536 MB 门槛，按实测重设门槛并在记录中说明，不得默默放宽。

### 代码接口（最小 diff，业务侧仅 `soap.py`）
- 拆 `get_orthogonal_matrix_QR` 为 `_qr_plan`（默认流，返回每因子 `(ind, sort_idx, power_iter)`）、`_qr_finish`（对 plan 逐个 `linalg.qr`）、`_qr_install`（重放 `index_select` + 写 `state['Q']`）。三者串行调用即逐位等价于现函数，k=0 时保持原路径。
- 周期分支：k>0 且合同内（FP32、merge_dims=False、非 channels_last-4D、shape 在 23 类白名单）时走 plan→侧流 finish→挂 pending(event, target=step+k)；否则同步。
- 安装检查点放在 `_step_foreach_chunk` 使用 `project` 之前；`state_dict()` 与 save 前强制 flush（event.synchronize + install），保证 checkpoint schema 与同步版同构。

### 静态包与局部门禁
- 镜像 STEP-216 布局：policy/gate/controller/两 shell/单测/source contract，tool-root 在业务仓库外（沿用 STEP-220 已验证 host-controller 布局），PGID-only 清理。
- 用 STEP-215-E 同款真实 checkpoint stateful adapter：双周期 + save/load/resume；断言 k=0 路径与现 HEAD 逐位一致、k=4 路径 pending 不重叠、flush 后 state_dict 同构、exp_avg_sq 置换重放顺序正确。
