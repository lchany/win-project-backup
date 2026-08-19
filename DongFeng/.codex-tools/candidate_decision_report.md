# GPU 合同对齐单次全阶段 Profiling：TopN 决策报告

## 1. 决策口径

- 数据只来自本轮唯一一次 rank0、`with_stack=True`、`record_shapes=True` 的连续窗口；Profiler Step 23～26 对应训练 Step 24～27。
- 训练 Step 24 是稳定 SOAP 周期步，Step 25～27 是稳定普通步；窗口位于动态 loss scale 恢复、首轮编译和数据管线热身之后。
- kernel device self time 用于 TopN，调用栈用于模块/源码归因；带栈 profile 的 step wall time 不作为吞吐结果。
- 最终性能仍采用 profiler-off 30-step 基线：普通稳定步 NPU/GPU 吞吐比约 0.700，包含 SOAP 的完整稳定周期约 0.510。

## 2. 分阶段结论

- 稳定 SOAP 周期步：QR 为绝对主项，纯 kernel device self time 约 22.897 s；调用栈落在 `SOAP.update_preconditioner`。
- 稳定普通步：每步稳定 TopN 依次为 MSDA backward（约 186.7 ms）、Conv2D（约 133.2 ms）、ViewCopy（约 95.9 ms）、BatchMatMul（约 82.9 ms）、MSDA forward（约 81.8 ms）、Conv TransData（约 67 ms）。
- 通信：AI Core 与 HCCL 的重叠率约 80.51%，不是当前首要瓶颈。
- wait-anchor：Index、Fill、Eq、Broadcast、Zero/MemSet 等包含显著排队/同步等待；不把 wait time 当成算子自身可回收计算量。

## 3. TopN 候选矩阵

| 优先级 | 对象 | 原因与源码归因 | 可考虑的优化方式 | 本轮决定 |
|---|---|---|---|---|
| 1 | SOAP QR | 周期步约 22.897 s，来自 `soap.py:update_preconditioner`，决定完整周期性能 | 只能接受与当前 QR、排序、符号及 optimizer state 完全等价的实现 | 关闭。固定客户软件版本下没有已证明状态等价且更快的实现；历史 batching/out-buffer/multistream/block/版本方案均无合格候选，不能以性能换优化器语义 |
| 2 | MSDA backward/forward | 普通步第一和第五；栈分别落在 MSDA backward 与 `spetr3d.py` 前向 | 算子融合/重写必须先做真实 shape 输出和梯度 exact/容差门禁 | 不重复开启。当前 HEAD 已包含 DrivingSDK MSDA 优化，新的 profile 是优化后残余成本，没有发现新的独立冗余边界 |
| 3 | Conv2D/Conv backward | 普通步稳定约 133 ms，来自图像/BEV 特征提取等必要主干计算 | 仅允许模块级冗余消除或固定合同下的等价融合 | 关闭。没有定位到可独立删除的业务冗余；改变 internal format、卷积算法或客户依赖版本均已拒绝/不适用 |
| 4 | ViewCopy/InplaceCopy | 普通步AI CPU ViewCopy约96 ms/2048次，另有AI Core ViewCopy约30～31 ms；host聚合代表栈为`spetr3d.py:1148→forward_rpn` | 只有在明确producer-consumer边界且不改变alias/stride时才可消除 | 关闭。可明确定位的`spatial_features_2d_lidar_pvb.clone()`最大单次copy仅6.888 ms，低于门槛；2048次聚合缺逐调用唯一栈，不能把聚合总量伪装成单一可删功能，且其中可能与已拒绝point_sampling边界重叠 |
| 5 | Matmul/BatchMatMul | 每普通步约82.9 ms的是单次`aclnnMatmul`，栈为`spetr3d.py:1182→BEVFormer point_sampling`；1284处`aclnnBatchMatMul`约280次但仅16～21 ms/步 | 已验证的表达包括显式repeat+matmul、broadcast/expand matmul和packed BMM | 关闭。broadcast虽节省约552 MB却从83.394 ms变慢到207.551 ms；packed BMM函数门禁83.410→1.856 ms且8/8 exact，但两轮正式30-step重复发生新进程编译成本，复验全步+3.529%、吞吐-3.409%、普通均值+0.284%，已正式拒绝 |
| 6 | Conv TransData | 普通步约 67 ms，与卷积格式转换共现 | producer/consumer 同格式且全链 exact 时才可消除 | 关闭。客户固定环境下 internal-format 路径已正式拒绝，不能修改 CANN/torch_npu/依赖版本 |
| 7 | Index/Nonzero/Unique/ReduceSum | 单项 device self 可见，但 Index 还带大量 wait；主要归因于 MapTR/目标生成路径 | 先区分真实计算与等待，再做固定顺序、重复索引、梯度语义门禁 | 不重开。相关 MapTR Unique/Index/Nonzero 路径已被既有提交覆盖或在正式 A/B 中拒绝；本次没有新的独立热点边界 |
| 降级 | Fill/Eq/Broadcast/Zero/MemSet 等 wait-anchor | 总时长主要是等待上游或同步，不是自身 kernel 算力 | 应追溯上游供给和同步链，不直接“优化”锚点 | 不列入 TopN 候选，避免把等待误判为计算收益 |

## 4. 结论与下一门禁

本次唯一 profile 复现了既有优化后残余瓶颈，但没有产生一个同时满足“客户固定环境、状态/功能等价、独立可回退、预期收益超过扰动”的新代码候选。因此本轮不为了制造优化提交而改代码，也不启动无依据的 8 卡 A/B。

若后续出现新的独立实现，必须按同一顺序执行：真实 shape 算子输出/梯度 A/B → 唯一容器中的 8 NPU profiler-off 短训练 A/B → 同一 checkpoint、同一测试集、同一样本顺序的功能/任务指标比较；三层门禁均通过且端到端稳定净收益成立后，才允许以 `【npu性能优化】<具体对象与动作>` 提交。
