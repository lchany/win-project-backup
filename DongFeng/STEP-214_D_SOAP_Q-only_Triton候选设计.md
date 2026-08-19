# STEP-214-D：SOAP QR 最小 Q-only Triton 候选设计

## 裁决摘要

当前只完成 CPU/源码设计，未编译、未调用 NPU、未训练、未 profiling，也未修改业务代码。Householder Q-only 在数学上可以实现，并可避免把 `R` 作为最终输出持久化；但它不能省略求反射子和更新尾矩阵所需的三角化工作，且自定义 Triton 的分块、归约顺序、符号选择与固定 SDK 的 `aclnnLinalgQr` 内部实现均不透明，无法预先证明 raw `Q` 逐位一致。因此当前裁决为：

`NO_GO_FORMAL_RAW_Q_EQUIVALENCE_UNPROVEN_DESIGN_ONLY`

该裁决不放宽项目规则。后续即使获准做最小 NPU 机制测试，只要任一 raw-Q 逐位门禁失败，就必须停止，不能用列符号对齐、正交残差、重构误差或短跑 loss 接近替代。

## 唯一允许的业务边界

权威源码为 `ascend_npu_optimize@f922c3897255` 的 `projects/mmdet3d_plugin/optimizers/soap.py`，文件 SHA256 为 `0e49429dbca9d9a2546c29f54e79639265f7468703ba4b36fa3b3796861a1077`。候选只能把：

```python
Q, _ = torch.linalg.qr(power_iter)
```

替换为：

```python
Q = q_only_qr(power_iter)
```

`diag(o.T @ m @ o)`、descending stable argsort、`exp_avg_sq`/`o` 同序重排、`power_iter = m @ o`、原 dtype 转换、`Q` 和 `exp_avg_sq` state 写回均保持原样。不得把排序、幂迭代、跨 shape batching或其他 consumer 聚合进候选。

## Shape、频次与状态合同

历史完整 24 类为：`1(106), 3(30), 4(6), 7(37), 8(1), 11(1), 22(1), 32(4), 40(9), 64(28), 96(3), 120(1), 128(18), 160(1), 192(32), 220(4), 256(181), 352(1), 440(4), 512(43), 768(22), 1024(6), 2560(8), 5120(4)`。当前 one-sided 路径删除 4 个 5120 和 4 个不必要的 2560，稳定 SOAP 周期仍有 543 次 QR，其中 2560 为 4 次。

单次 QR 合同要求输入 shape/stride/dtype、finite/NaN/Inf/异常行为和 raw `Q` 逐位一致。完整 SOAP 合同要求至少连续两个 QR 周期，以及周期之间 checkpoint/resume 后的 `Q/GG/exp_avg/exp_avg_sq/state.step/参数`逐位一致，state_dict 的 key、shape、dtype、device 不变。24 类门禁必须保留，以覆盖旧 checkpoint/配置回退边界。

## 最小 Householder Q-only 数据流

候选不使用 Gram-Schmidt、Cholesky-QR 或 TSQR。只允许标准 FP32 Householder 路径：

1. 把输入复制到工作矩阵 `Awork`；按列或小 panel 计算 `norm2`、符号规范、反射向量 `v` 与 `tau`。
2. 将反射子应用到尾矩阵：`A_tail -= v * (tau * v.T @ A_tail)`。上三角 `R` 仍会在 `Awork` 内形成，但不另建、不返回最终 `R` tensor。
3. 保存紧凑反射子（`Awork` 下三角和 `tau`），从单位阵开始按反向次序应用反射子，物化最终 `Q`。
4. 只返回连续 FP32 `Q`；wrapper 保持当前 dtype 转换和 optimizer state 写回。

对于 2560/5120，单个方阵 FP32 分别约 25/100 MiB，远超 Triton-Ascend 文档给出的单 kernel tensor 总量 96 KiB（关闭 double buffer 时 192 KiB）限制，因此不可能做单 program、全片上持久实现。必须 tile 到 GM，并以同 stream 的多 kernel/panel 边界保证阶段顺序。

## Triton-Ascend 3.2.0rc4 表达能力与限制

已隔离 wheel 暴露 `load/store/arange/program_id/sum/max/sqrt/rsqrt/where/dot/trans/atomic_add/make_block_ptr/advance`，FP32 `tl.dot` 可选 `ieee`；因此反射子归约、panel 外积更新和 Q 更新在 DSL 层面可表达。排序应继续留在现有 PyTorch 边界，因为该版本官方 API 表明确 `sort` 不受支持。

扩展同步接口的真实签名为 `sync_block_all(mode, event_id)`，其中 mode 仅为 `all_cube/all_vector/all`；`sync_block_set/wait`只允许 `cube -> vector` 或 `vector -> cube`。官方仅称其为需谨慎使用的“核间同步指令”，没有给出跨任意 program 的全局内存可见性、调度和无死锁合同。因此当前不能把它当作跨 panel 的 grid-wide barrier。最小安全设计采用多 kernel host loop；单 kernel persistent QR 仅能作为未来独立机制研究，不能作为当前正确性假设。

## 建议的仓库外原型文件边界

- `q_only_qr.py`：shape/dtype/stride 检查、workspace 分配、默认关闭的调用封装及当前 `torch.linalg.qr` oracle。
- `q_only_householder.py`：panel reflector、trailing update、反向生成 Q 三类 Triton kernel；不包含 SOAP 排序、state 或 checkpoint 逻辑。
- `test_q_only_qr_contract.py`：24 类单 QR、异常输入、连续 SOAP 状态和 resume 合同。
- `benchmark_q_only_qr.py`：按当前 543 次 shape/count 加权的独立周期性能；不以单个大 shape 或 profiler wait 代替周期净收益。

在任何正式通过前，这些文件只能位于仓库外诊断目录。若全部门禁最终通过，业务侧仅形成一个可独立回退的功能提交，提交信息使用 `【npu性能优化】SOAP QR仅生成Q`；当前不创建该提交。

## 分级门禁（不改变正式规则）

1. **G0，CPU/源码门禁**：检查算法索引、workspace 上限、24 类 shape dispatch、state schema 和异常合同；用高精度/CPU QR 只做数学诊断。G0 不能证明 NPU raw-Q 等价。
2. **G1，未来最小 NPU 编译门禁**：仅在主任务明确释放资源后，独立编译 tiny shape；确认 backend、无 CPU fallback、无越界和确定性。当前禁止执行。
3. **G2，raw-Q 早停门禁**：按小到大覆盖 24 类、多个普通/近秩亏/秩亏/零/NaN/Inf 输入。候选 raw `Q` 必须逐位等于当前 `aclnnLinalgQr`；任一元素不同立即正式拒绝。符号对齐和容差指标只记录诊断，不能继续性能 A/B。
4. **G3，SOAP 状态门禁**：连续至少两个 QR 周期以及中途 checkpoint/resume，完整状态、参数和 schema 逐位一致。
5. **G4，独立加权性能门禁**：当前 QR 每 10 step 一次，故要贡献 `>22.7 ms/step`，每个 543-call QR 周期的净节省必须严格 `>227 ms`，且置信区间下界也越线。当前 QR device time 为 `22641.384 ms/cycle`，即至少约 1.01x 的周期加速；必须计入 workspace、launch、同步与 wrapper 成本。
6. **G5，正式业务门禁**：仅 G2～G4 全部通过后，才允许 8 卡短训、稳定长窗、876/resume、同 checkpoint/测试集和 GPU 基线比较。loss/grad/state/最终功能合同保持现有规则，不因自定义算子放宽。

## 性能上限解释

当前 543 次 QR 若 `R` 为完整方阵，单周期仅约 `285,545,768 bytes`（0.266 GiB）最终输出；其有效上三角载荷约 0.133 GiB。即使按 200～1000 GB/s 粗略带宽下界，单纯少写这一个输出只约 1.33～0.27 ms/cycle，远低于 227 ms/cycle 门槛。候选若有收益，必须主要来自把当前 `QrAiCPU` 的完整 QR 改为高效 AI Core tiled Householder，而不能把“省 R 写出”宣称为越线收益。与此同时，这种实现变化正是 raw-Q 舍入顺序最难保持的来源。

## 最终结论

Q-only Householder 的数学和 DSL 表达性成立，但严格 raw-Q/state 等价尚无证明，且单纯省略 R 输出的理论收益不够。当前只能保留为仓库外、默认关闭、按 raw-Q 早停的研究设计；不得进入业务改码、8 卡 A/B 或采用流程。
