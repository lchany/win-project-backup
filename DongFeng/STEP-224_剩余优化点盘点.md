# STEP-224：当前基线下剩余优化点盘点（只读）

## 当前基线
- HEAD：`2a2aa0f`（pin）+ 启用时 `SOAP_STALE_Q_K=4`（`2846401`）
- 100-step 证据：全窗 **24.15 samples/s**；late≥10 **≈27.33 samples/s**
- GPU 参考（历史同口径窗口）：约 **27.6 samples/s**
- 粗比值：全窗约 **0.87:1**；稳态窗口约 **0.99:1**（已接近 1:1）

## 已关闭 / 不宜重开（无新证据前）
| 方向 | 状态 |
|---|---|
| SOAP Brockett / 自定义 QR | 数值 REJECT |
| QR batching / 多流缩短 Step10 | 关闭 |
| 无栈 Level0 普通步拼 gap | STEP-222 `NO_GO` |
| TASK_QUEUE=2 / COMBINED / CPU affinity | 回归或无合同 |
| Conv/MatMul HF32 | 数值扰动 + 低收益 |
| 全局 internal format | 回归 |
| NPUGraph | 无净收益 |
| MSDA 项目侧改 / SDK 升级 | 冻结环境禁止 |
| 无条件 pin（旧 SOAP 成本） | 已在新基线重测并提交，勿再按旧 REJECT 回退 |

## 仍可能有价值的点（按优先级）

### A. 新基线 Level0 普通步 profile（证据刷新）— 中优先
- **为何**：STEP-222 采在 pin 之前；pin 已改 H2D/Preparing 结构，旧 underfeed 归因可能过期。
- **做法**：一次 rank0 Level0、无栈无 shape、覆盖稳态普通步；只筛 **单一等价边界 >22.7ms**。
- **风险**：很可能再次 `NO_GO`；若无边界则停止改码，不放宽门禁。
- **功能门禁**：同 STEP-222（默认同输入同输出）。

### B. `expandable_segments` 正式化 — 低～中优先
- A/B 夹具已设 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`；`tools/ddp_train.sh` 工作树有未提交改动。
- R6 曾判“无 OOM 不必开”，但 stale-Q/pin 正式跑一直带着它。
- **下一步**：确认生产入口是否已生效；若仅夹具有、正式脚本无，做**单变量**短 A/B 后再决定是否独立小 commit。不与其它变量捆绑。

### C. SOAP `k` 细调（2/3/8）— 需语义授权
- 现固定 k=4；改 k 改变 Q 陈旧相位，属 stale-Q 同类语义轨道。
- 无用户授权不做；预期是边缘收益，不是第二颗银弹。

### D. 局部 TransData / layout — 低优先
- 历史有 kernel 暴露，但从无唯一可证明等价、>22.7ms 源码边界。
- 仅当 A 的新 profile 钉死**单一** layout 边界时再开。

### E. 非性能但影响验收
- 测试集 / 推理指标对齐（迁移精度故事）；与吞吐 1:1 分列。

## 不建议再投入的“看起来热”的点
- 再抠 Index/Nonzero/Unique、grad clip fused、DataLoader worker 数（已 8+prefetch3）
- 再开 CPU-NUMA / TASK_QUEUE / HF32 / NPUGraph
- 把分散 host gap 加总冒充单一优化

## 建议决策
1. **若目标是报表 1:1（全窗 samples/s）**：先做 **A**（新基线 Level0）；有边界再改，无边界则承认剩余主要是分散 host/下发，短期不可提交级优化。
2. **若接受稳态已≈1:1**：优化主线可收口，转验收（评测）与文档；B 仅作启动脚本卫生项。
3. **默认不自动占卡**，除非明确要跑 A 或 B。

## STEP-225 执行结果（2026-08-16）
- A：Level0 在 `2a2aa0f`+k4+pin 下完成 → `NO_GO_NO_UNIQUE_EQUIVALENT_BOUNDARY_ABOVE_22P7MS`（单次空洞不复现、无栈）。
- B：`expandable_segments` 30-step → 吞吐 +1.73%、peak -144MiB → `PASS_EXPANDABLE_BETTER`；已提交 `fa95a2a`。
- 结论：当前无更多可独立验证的业务源码级性能提交；剩余全窗缺口以分散 host/下发为主。

## STEP-226 收尾（2026-08-16）
- 用户指令「按照收尾来做」→ 性能主线 `PERF_MAINLINE_CLOSED`；文档固化于 `STEP-226_性能主线收尾验收.md` 与 `最终性能优化报告.md` §0。
- 评测：容器 `ortools=False`，canonical 测试集评测未跑、不伪造通过。
- 不再占卡做性能挖潜；push / OR-Tools 评测环境另指令。
