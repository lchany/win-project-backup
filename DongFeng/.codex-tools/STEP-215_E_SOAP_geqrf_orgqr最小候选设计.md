# STEP-215-E：SOAP `geqrf + orgqr` 最小候选设计

## 当前裁决

仅完成仓库外设计和补丁草案；未修改权威业务文件、未运行 NPU、训练或 profiling。候选只有一个可接受的源码边界：在 `SOAP.get_orthogonal_matrix_QR` 中把

```python
Q, _ = torch.linalg.qr(power_iter)
```

替换为：

```python
packed, tau = torch.geqrf(power_iter)
Q = torch.orgqr(packed, tau)
```

草案文件为 `.codex-tools/step215_soap_geqrf_orgqr_candidate.patch`。只有 24 类真实 shape 数值门禁全部通过且当前 543 次调用的加权净节省严格大于 `227 ms/cycle`，才允许把该草案复制为业务候选。

## 权威源码定位与调用点

- 既有权威只读审计对象为 `ascend_npu_optimize@f922c3897255`，目标文件 `projects/mmdet3d_plugin/optimizers/soap.py`，当时 SHA256 为 `0e49429dbca9d9a2546c29f54e79639265f7468703ba4b36fa3b3796861a1077`。
- 历史权威源码和当前本地保存的原始补丁都把唯一业务 QR 调用定位到 `get_orthogonal_matrix_QR` 的 `power_iter = m @ o` 后。该函数由 `update_preconditioner` 在首次 Q 初始化及 `state.step % precondition_frequency == 0` 时调用。
- 补丁按文件和上下文精确命中这一处，不会替换项目中其他模块的 QR。由于本地 Paramiko 依赖目录 ACL 不可读，本轮 fresh 远端 project-wide `rg` 在建立连接前失败；因此“当前仍只有一个调用点”以既有同 SHA 权威审计为依据，实际应用前必须再次断言目标文件 SHA、该上下文恰好一次以及全项目 QR 调用清单。

## 保持不变的语义

候选不移动、不融合下列任一步骤：

1. `est_eig = diag(o.T @ m @ o)`；
2. descending、`stable=True` 的 `argsort`；
3. `exp_avg_sq.index_select(ind, sort_idx)` 与 `o.index_select(1, sort_idx)`；
4. `power_iter = m @ o`；
5. Q 转回 `precond_list[ind].dtype` 的逻辑；
6. `state['exp_avg_sq']` 重排写回、最终 `state['Q']` 持久化、更新频率和所有 parameter-group 字段。

`packed` 和 `tau` 只是在函数栈内存活的临时 tensor，不写入 optimizer，不进入 `state_dict`。`torch.orgqr(packed, tau)` 返回与方阵 `power_iter` 同 shape 的显式 Q；已有下游只消费 Q。`packed` 的上三角虽可解释为 R，但业务不消费 R，候选不得为了诊断在正式路径执行 `torch.triu(packed)`。

因此补丁自身不新增或删除任何 state key，不改变已有 state tensor 的 shape、dtype 或 device。应用门禁仍须逐项验证这一结论，不能只凭静态设计通过。

## `out`、alias 与 autograd 边界

- 现行调用没有使用 `out=`；候选也不引入 out buffer。`geqrf`、`orgqr` 的返回均为新 tensor，不能假定与 `power_iter` 或 Q state 共享 storage。
- `packed` 同时存放 R 和 Householder 反射子；`orgqr` 必须读取完整 packed tensor，不能在调用前对其做 `triu_`、复用为 Q buffer或提前释放。
- `torch.linalg.qr(..., reduced)`公开为可微操作，而低层 `geqrf/orgqr` 的 autograd 路径不应被假设完全相同。当前候选位于 `SOAP.step` 的 `torch.no_grad()` optimizer 调用链，Q 是 optimizer state而非模型前向图节点，所以正式门禁要在 QR 边界记录 `torch.is_grad_enabled() == False`、`power_iter.requires_grad == False`、`Q.grad_fn is None`。若未来从 grad-enabled 路径直接调用 `get_orthogonal_matrix_QR`，该候选不具备语义承诺，必须回退 baseline。

PyTorch 2.7 官方依据：`geqrf`返回 packed reflectors 与 `tau`，`orgqr`是 `linalg.householder_product` 的别名；`linalg.qr` 的有效分解可因 R 对角符号不同而非唯一。参考：

- https://docs.pytorch.org/docs/2.7/generated/torch.linalg.qr.html
- https://docs.pytorch.org/docs/2.7/generated/torch.orgqr.html
- https://docs.pytorch.org/docs/stable/generated/torch.geqrf.html

## finite 与异常合同

补丁不增加 `isfinite`、同步或 Python 分支，正常有限输入直接沿用两原语的自然行为。尽管如此，两条实现对异常输入、秩亏输入的成功/失败和非有限传播可能不同，不能由数学等价推断。仓库外门禁必须至少覆盖：

- 正常 full-rank、近秩亏、严格秩亏、全零方阵；
- 单个及多处 `NaN`、`+Inf`、`-Inf`；
- 所有 24 类方阵维度，FP32、真实 contiguous/stride；
- baseline/candidate 是否都成功或都抛异常；若抛异常，异常类型必须一致；若成功，Q 的 shape/dtype/device、finite mask及数值门禁必须满足冻结合同。

PyTorch 官方明确说明秩亏 QR 可能不抛错但结果不正确，因此不得把“没有 RuntimeError”当通过。任何一侧 hang、AICPU/CPU fallback变化、异常类型变化或 finite mask变化均立即拒绝。错误消息文本可记录，不以跨后端容易变化的完整字符串作为硬门禁。

## 仓库外双 QR 周期 harness 设计

该阶段只在 24-shape 门禁通过后实施，且不先修改业务文件。

### 1. 源码和状态入口

- 启动时断言 branch/HEAD、`soap.py` SHA、唯一替换上下文、正确容器和 8 rank/后8 NPU；baseline 与 candidate 都继承同一个 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`，`TASK_QUEUE_ENABLE`保持共同缺席。
- 在远端原位加载同一个已保留 checkpoint，不下载。按 param-group 顺序构造两条相同 SOAP optimizer 轨迹；每条从同一参数、optimizer state、RNG snapshot开始。
- 为避免业务改码，candidate 轨迹仅在 `optimizer.step()` 的动态作用域内临时包装 `torch.linalg.qr`：只接受 `mode='reduced'`、`out is None`，内部返回 `(torch.orgqr(*torch.geqrf(A)), None)`；退出作用域立即恢复原函数。包装器统计调用次数，并断言 QR 边界处 grad disabled。baseline 不包装。此机制只用于状态门禁，不作为最终业务实现。

### 2. 三轨自噪声与输入合同

顺序运行 `baseline-A`、`baseline-B`、`candidate`，不能并发占卡。三轨从同一 snapshot开始，按 parameter index和有效 optimizer step生成完全相同、可重放的 FP32 gradient；每步先记录 gradient 摘要和 RNG digest，确保输入一致。baseline 双跑用于估计固定栈自噪声，candidate 阈值不能因此超过硬上限。

不用合成 loss 冒充训练 loss；optimizer-only harness只验证状态传播。真实 loss、grad、dynamic loss scale 与 overflow/skip 相位留给后续 30-step 正式 A/B。

### 3. 周期检测和比较

不硬编码训练 iteration；以包装器/observer 实际捕获到非零 QR 调用的 optimizer step为周期边界，直到连续捕获两个真实 QR 周期。每周期保存：

- QR 调用数、各 shape/count、`state.step`和排序索引 digest；
- state_dict key、Python type、tensor shape/dtype/device；
- 参数和 `Q/GG/exp_avg/exp_avg_sq` 的 finite、norm、global relative-L2及逐 tensor最坏 NRMSE；
- candidate 的 `packed/tau`只统计峰值显存和生命周期，不进入 state schema；
- 8 rank各自结果和最坏值，不能只报 rank0或均值。

硬门禁沿用 STEP-215-B：`sort_idx/state.step/schema` exact；Q 逐 tensor NRMSE `<= min(5e-5, max(1e-5, 2×baseline自差))`；其他持久 state和参数 global relative-L2 `<= min(1e-4, max(1e-5, 2×baseline自差))`。所有 tensor必须 finite。任一误差从第一到第二周期单调放大并触及阈值、排序变化、调用数/shape变化或 rank离群立即拒绝。

### 4. resume 支路

第一个 QR 周期后各轨迹分别保存 checkpoint并记录 SHA/schema；销毁 optimizer和参数，重新构造后 load，再重放完全相同的余下 gradient/RNG序列直到第二周期。分别比较：

1. baseline连续 vs baseline resume；
2. candidate连续 vs candidate resume；
3. baseline vs candidate。

resume差异不得超过对应不中断差异的2倍，且仍受 `5e-5/1e-4`硬上限约束。checkpoint schema、step、排序、finite必须 exact。

## 应用与回退

通过局部shape、双周期和resume后，才将草案作为唯一业务改动应用；执行 `py_compile`、`git diff --check`、精确diff与全项目调用点复核。随后才允许30-step和876-step。若最终采用，只形成一个可独立回退的提交：`【npu性能优化】SOAP QR改用显式Householder原语`。任一门禁失败，精确恢复这一处业务hunk，保留仓库外证据，不触碰用户要求保留的 profiling 原始数据。

## 双周期执行实现状态

仓库外执行框架已实现为：

- `step215_e_soap_two_cycle_gate.py`：校验 `soap.py` SHA 和唯一 AST 调用上下文；串行执行 baseline-A、baseline-B、candidate；三轨逐步重新生成并校验同一 gradient digest；仅在目标 SOAP 函数动态替换 QR；以实际非零 QR 调用识别两个周期；首周期分别保存/load，第二周期同时推进 continuous/resume；比较 schema、stable sort digest、`state.step`与finite。candidate每周期的Q有效上限为`min(5e-5,max(1e-5,2×baseline-A/B自差))`，其余 state/parameters 为`min(1e-4,max(1e-5,2×baseline-A/B自差))`；candidate continuous/resume另以baseline-A和baseline-B各自continuous/resume自差的最大值按同式校准，硬上限永不放宽。
- `step215_e_run_inside_container.sh`：冻结 `mapqr-leicheng` 内 world8、后8逻辑 NPU、allocator 开关、TQ2 缺席、超时与 live `npu-smi` controller。
- `step215_e_host_launch_contract.sh`：参数化 repo/config/checkpoint/adapter/output/port/SHA，并在唯一正确容器和 active=0 后才进入 runner。
- `step215_e_real_soap_adapter_TEMPLATE.py`：列出真实 optimizer 构造、checkpoint、可重放 gradient、完整 state view 和 resume 所需的10项 readiness；模板全部为 false 且所有方法抛 `NotImplementedError`。

本地没有远端权威源码、config 和 checkpoint，不能诚实实现真实 SOAP optimizer adapter。因此当前状态为 `FAIL_CLOSED_SCAFFOLD_IMPLEMENTED_NOT_RUNTIME_READY`，不是双周期门禁 PASS。只有在远端原位按权威源码填写 adapter、将全部 readiness 逐项验证为 true 后，launcher 才允许执行；不得用合成 optimizer 或合成 loss 补齐。
