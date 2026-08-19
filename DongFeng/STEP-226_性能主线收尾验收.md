# STEP-226：性能主线收尾验收（2026-08-16）

## 裁决
`PERF_MAINLINE_CLOSED` — 固定环境内无可继续独立提交的等价源码级优化；吞吐主线按当前证据收口。测试集 canonical 评测仍受外部依赖阻塞，不伪造通过。

## HEAD 与启用合同
| 项 | 值 |
|---|---|
| 分支 | `ascend_npu_optimize`（未 push） |
| HEAD | `fa95a2a` |
| 容器 | 仅 `mapqr-leicheng`，后 8 卡，8 rank |
| 本轮收口提交链 | `2846401` stale-Q → `2a2aa0f` pin → `fa95a2a` expandable_segments |

生产启用（须同时满足）：

```bash
export SOAP_STALE_Q_K=4   # 默认 0=同步回退；证据合同为 k=4
# tools/ddp_train.sh 已默认：PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# 配置侧 pin_memory=True；DataContainer.pin_memory 已补齐生效
```

回退：不设或 `SOAP_STALE_Q_K=0` 即恢复同步 QR；allocator 行可从 `ddp_train.sh` 删除独立回退。

## 吞吐结论（相对 GPU 参考日志，非严格同数据合同）
| 窗口 | NPU（STEP-223-B，k=4+pin） | GPU 参考 | 粗比值 |
|---|---:|---:|---:|
| 100-step 全窗 | ~24.15 samples/s | ~27.6 | ~0.87:1 |
| 稳态 step≥10 | ~27.33 samples/s | ~29 | ~0.94–0.99:1 |

历史 `f922c38` 876-step 公共窗约 0.55:1 已被 SOAP stale-Q + pin 大幅收窄；全窗仍未宣称严格 1:1。剩余缺口以分散 host/下发为主，STEP-225 Level0 无单一 >22.7ms 等价边界。

## 正确性 / 收敛已具备证据（不重复占卡）
- stale-Q：Stage A–D（含 300-step）通过后提交 `2846401`
- pin：100-step e2e PASS 后提交 `2a2aa0f`
- expandable：30-step PASS 后提交 `fa95a2a`
- 更早 MSDA 链：876-step、checkpoint/resume、512-sample 推理相对父提交无数量级偏离

## 评测验收（未完成 — 外部阻塞）
| 阻塞 | 状态 |
|---|---|
| 容器无 `ortools`，canonical `eval_RG.py` 不可用 | 禁止安装依赖，未跑完整测试集 |
| 活动 dataset 不接受历史 `lidar_type`；评测数据身份难唯一恢复 | STEP-185 / 后续审计已记录 |
| 客户兼容 OR-Tools 环境或既有评测机 | 未提供 |

补齐条件：客户提供兼容评测环境或授权在隔离环境安装 OR-Tools 后，按冻结 config/checkpoint/数据合同跑完整样本指标；**不需要再改当前业务性能代码**。

## 明确不做
- 不再开 Level0/拼 gap、k 扫描（除非新证据或显式语义授权）
- 不重开 Brockett / QR 替换 / NPUGraph / HF32 / TASK_QUEUE 等已关闭项
- 不 push（须另指令）
- 不把 `fusion_result.json` / 大 trace / diagnostics 纳入提交

## 资源核验（收尾时）
- 后 8 卡训练 Python 进程：0
- 业务 tracked 工作树：仅已知 `fusion_result.json` 脏；无未提交性能 diff
