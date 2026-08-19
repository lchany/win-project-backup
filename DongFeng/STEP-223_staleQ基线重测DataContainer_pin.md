# STEP-223：stale-Q 基线下重测 DataContainer pin_memory

## 为什么可以重开
- `f922c38` 上无条件 pin 正式 30-step：普通步约 -9.65%，但 SOAP +6.70%，端到端吞吐 -3.36% → `REJECT_NO_COMMIT`。
- 现基线为 `2846401` + `SOAP_STALE_Q_K=4`：SOAP 周期从 ~33s 降至 ~6s，同样的 SOAP 相对惩罚绝对值大幅缩小；普通步收益更可能转化为端到端净收益。
- 功能：历史门禁已证明 tensor 值/shape/dtype/stride **逐位一致**；本轮不申请新语义，只重测性能。

## 单变量
- 对照：HEAD `2846401` + `SOAP_STALE_Q_K=4` + 无 `DataContainer.pin_memory`
- 候选：同上 + 仓库 `mmcv/parallel/data_container.py` 补齐 `pin_memory()`（使已有 `pin_memory=True` 生效）

## 门禁
1. 静态：训练 PYTHONPATH 下导入的是仓库 mmcv；`py_compile`；短探针 nonempty tensor `is_pinned()`。
2. 8 卡 30-step×2，后 8 卡，GPU 对齐合同，profiler-off。
3. 通过：双方 exit0、loss/grad 有限；候选 **端到端吞吐（samples/s）严格优于** 对照；峰值显存增幅可记录。
4. 失败：精确恢复文件，记 `REJECT_STILL_NO_E2E_UNDER_STALE_Q`，不 commit。

## 30-step 结果（2026-08-16）
- 产物：`diagnostics/step223_pin_staleq_k4_30step_8npu_20260816T023435/`
- 裁决：`REJECT_STILL_NO_E2E_UNDER_STALE_Q`（吞吐 **-14.83%**，全窗 mean +17.4%）。
- 细节：冷启动候选 Iter1/2 明显变慢；**step10–29** 候选 mean 4.599s vs 基线 5.721s（吞吐约 27.8 vs 22.4）。`data_container.py` 已恢复权威 SHA。
- 跟进：STEP-223-B 开 100-step 同合同 A/B，检验摊销后全窗是否净收益（功能仍为等价 pin）。

## 100-step 结果 STEP-223-B（2026-08-16）
- 产物：`diagnostics/step223b_pin_staleq_k4_100step_8npu_20260816T025510/`
- 裁决：`PASS_E2E_THROUGHPUT`
- 全窗吞吐 **20.200 → 24.155 samples/s（+19.58%）**；full mean 6.337→5.299 s（-16.37%）
- late≥10：**22.370 → 27.334 samples/s（+22.19%）**，接近 GPU 参考 ~27.6
- 普通/SOAP 均值均下降约 17–18%；peak mem +3 MiB；双方 100/100、loss 有限
- 工作树：A/B 后已恢复再装入 `pin_memory`，待用户要求时单文件 commit：`【npu性能优化】DataContainer补齐pin_memory使页锁定生效`
