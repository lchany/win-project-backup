# STEP-222 P1 普通步优化方案（功能等价硬门禁）

## 目标
在已合入的 stale-Q（`2846401`，启用 `SOAP_STALE_Q_K=4`）之上，压缩普通步相对 GPU 的约 1.55 s 缺口，推进 NPU:GPU 从约 0.77:1 向 1:1。

## 功能硬门禁（用户 2026-08-15 明确）
任何候选在宣称通过前必须证明：

1. **同输入 → 同输出**（默认）：在固定 seed/合同下，与基线对比模型输出、loss（或约定 fingerprint）在允许偏差内；默认目标为**逐位一致或既有项目数值门禁内**。
2. **不允许静默语义漂移**：若改动会改变数学顺序、舍入轨迹、随机性、optimizer 状态更新时机，必须事先声明为新语义轨道并获与 stale-Q 同级授权；否则一律按等价路径做，通不过即关闭。
3. **门禁链不得跳过**：仓库外/静态 fail-closed → 机制微基准（若适用）→ 8 卡短训 A/B（loss/grad 有限 + 输出/状态合同）→ 再加长窗。任一层失败回退，不提交。
4. **基线定义**：P1 的对照基线 = `ascend_npu_optimize@2846401` + `SOAP_STALE_Q_K=4` + 既有 GPU 对齐合同（后 8 卡、batch16、seed0）。默认 k=0 仅作回退，不做 P1 主对照。

允许偏差仅限：
- 项目已成文的数值门（如既有 loss/grad 有限、动态 loss scale 早步行为）；
- 或用户事先书面批准的语义/容差（本轮 P1 **默认不申请**新语义容差）。

## 已关闭、禁止重开
NPUGraph / TASK_QUEUE=2 / 无条件 pin_memory / Conv·MatMul HF32 / MSDA 项目侧改实现 / 已拒绝的 mask·Unique·packed-BMM 等。STEP-196 曾因“无新 profile”关闭 underfeed；本轮**允许一次**新的 Level0 低扰动采集，但不得把分散 gap 拼成假单一边界。

## 执行顺序
1. **证据**：rank0 Level0、`with_stack=false`、`record_shapes=false`，覆盖稳定普通步（避开 SOAP 周期与冷启动）；原位解析 launch-gap / host idle；用后按规则删除 raw。
2. **筛选**：只保留“单一源码边界、状态等价可证明、理论 profiler-off 收益 >22.7 ms/step”的候选。
3. **实现**：最小 diff；优先消除冗余同步/重复下发/可证明的空拷贝等**等价**改动。
4. **验证**：同 checkpoint 或同 seed 短轨迹上比对输出/loss/关键 state；8 卡 30-step A/B；通过才准备 `【npu性能优化】…` 单 commit。

## 预期
P1 单项不确定能否吃满 1.55 s；GPU 侧也有 underfeed。若 profile 仍无法钉死单一等价边界，记录 `NO_GO` 并停止空想改码，不放宽功能门禁凑性能。

## 结果（2026-08-15）
- 采集：`diagnostics/step222_p1_level0_ordinary_k4_8npu_20260815T231452/`（`SOAP_STALE_Q_K=4`，wait22/warmup1/active2，无栈无 shape，26 step，exit0）。
- 裁决：`NO_GO_NO_UNIQUE_EQUIVALENT_BOUNDARY_ABOVE_22P7MS`。
- 要点：排除 QR 后仅 1 次 36.7ms 空洞且不复现、不可归因；分散簇禁止拼接；raw 已删，仅留 analysis 脱敏摘要。
- 行动：无代码变更、无 A/B、无 commit。
