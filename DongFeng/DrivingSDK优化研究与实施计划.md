# DrivingSDK 优化研究与实施计划

> 交接用途：交给另一个模型继续研究和实施。本文只定义实验队列和门禁，不代表所有候选都应修改。

## 1. 目标

在不改变客户模型功能、loss 语义和训练环境的前提下，继续优化 `ascend_npu_optimize` 分支的 Ascend NPU 训练性能。所有结论必须来自当前客户 batch16、8 NPU 真实调用链；“代码里存在”“DrivingSDK 提供了 API”或“修改后能运行”都不能单独证明优化成立。

## 2. 当前权威状态

- 远端正确容器：`mapqr-leicheng`。
- 当前分支：`ascend_npu_optimize`。
- 当前 HEAD：`bf9ed6e 【npu性能优化】TextLogger显存统计同步降频`，工作树干净，未 push。
- 环境冻结：PyTorch 2.7.1、torch_npu 2.7.1、CANN 8.3；不得升级、降级或重装。
- `mx_driving` 已安装，版本 `1.0.0+gitde13346`。
- 正式负载：8 rank，batch/rank=16，全局 batch=128，workers/rank=8，30 step，profiler-off。
- 当前 HEAD 的 30-step 数据：全步均值 9.313947 s，吞吐 13.742831 samples/s；普通 23 步 mean/median/P95=5.715218/5.778270/5.922613 s；SOAP 双步窗均值 34.530815 s；framework 峰值 26851 MiB/rank。

## 3. 不可违反的边界

1. 训练与 NPU 测试只能在完整名称为 `mapqr-leicheng` 的容器内执行，正式验证必须恰好 8 rank。
2. 每次实验前核验 Git HEAD/status、容器、8 卡可见性、`torch_npu`、`npu-smi`、端口和既有训练进程。
3. 不修改驱动、固件、CANN、PyTorch、torch_npu、Python 依赖或客户数据。
4. 一次只改一个独立机制；通过后只形成一个 `【npu性能优化】<对象>` commit；不 push。
5. 原始日志、profile、checkpoint 和客户数据只留远端，不下载到本地。
6. 所有操作追加到 `操作步骤.md`；失败候选也必须记录原因并恢复到干净 HEAD。
7. 当前阶段禁止恢复、放开、适配或专项测试任何已注释算子；该工作整体延期，只有用户以后再次明确启用时才重新制定独立计划。

## 4. 当前立即执行项（P0）

### R0：删除 MapTR 正负样本索引的冗余 `.unique()`

现有证据：

- 文件：`maptrv2_head_decoder.py:965～968`。
- 当前 batch16 profile 中两个 Unique 合计约 49.623 ms/step，调用量约 256 次/step。
- 旧表达式 `nonzero(...).squeeze(...).unique()` 与新表达式 `nonzero(...).squeeze(...)` 已完成 131,370 个 CPU case，值、顺序、shape、dtype 完全一致。
- 两行最小 patch 已做 `git apply --check`，但尚未应用到业务仓库。

执行顺序：

1. 再次确认后 8 卡空闲、正确容器、`bf9ed6e` clean。
2. 只应用两行替换，不顺带重写 mask、权重构造或 `torch.isin`。
3. 运行后 8 卡 NPU 函数级 exact 门禁，比较输出、dtype、shape、顺序以及涉及梯度时的梯度。
4. 运行客户 batch16、8 rank、30-step profiler-off A/B。
5. 检查 30 个 loss/grad 全有限；比较普通步、全步、吞吐、SOAP 窗和显存。
6. 收益超过运行波动且无回归时，提交：`【npu性能优化】MapTR正负样本冗余去重消除`；否则恢复文件且不提交。

## 5. 后续研究队列

### P1：优先研究，最可能形成下一项实验

#### R1：MSDA 实际实现与 DrivingSDK 融合实现对比

- 先用 profile/call stack 确认实际活跃的是项目自定义 NPU 扩展、MMCV 扩展还是其他路径，禁止按源码中的 CPU fallback 关键词推断。
- 当前迁移层会使 `torch.cuda.is_available()` 和 NPU tensor 的 `is_cuda` 条件成立，因此项目自定义 attention 目前并非已证实的 CPU reference fallback。
- 对“当前实际实现”与 `mx_driving.multi_scale_deformable_attn` 做同 shape、dtype、layout、mask 的 forward/backward 等价测试。
- 必须覆盖梯度、混合精度、动态 shape 和 DDP；等价后才做孤立微基准与 8 NPU A/B。
- 不能重复“只删除 `zeros_like` buffer”方向；该方向上限约 2.048 ms/step，已低于噪声门槛。

#### R2：MapTR target/GT 预处理的 host-bound 路径

- 从当前 profile 的 bubble 和 host stack 定位 decoder 内重复的 GT shift、采样、索引、deepcopy、Python 循环和 CPU/NPU 转换。
- 统计每个 batch、decoder layer、rank 的调用次数和输入规模，确认哪些结果可在 dataset/collate 阶段只计算一次。
- 优先设计“缓存不可变目标张量”或“前移到 DataLoader”的最小方案；不得改变数据增强随机性、样本顺序或标注语义。
- 验证同一输入下所有 target tensor bitwise/equivalent、loss/grad 有限，再做 8 NPU A/B。

### P2：有条件研究

#### R4：标准 MultiheadAttention 融合

- 仅当当前 profile 证明标准 MHA/SDPA 是稳定热点时，研究 TorchNPU `npu_fusion_attention`。
- 验证 QKV layout、head 维度、attention mask、dropout、scale、训练/推理模式和 backward；DrivingSDK 不是这个候选的唯一来源。

#### R5：BEV Pool 版本选择

- 当前项目已经使用部分 `mx_driving.bev_pool`；先确认活跃调用、版本和耗时，再比较 v1/v2/v3。
- 必须使用真实 ranks/intervals/shape 做 forward/backward 等价，不得只以 API 版本号判断新版本更快。

#### R6：运行时开关单变量 A/B

研究候选：`TASK_QUEUE_ENABLE=2`、`CPU_AFFINITY_CONF=1`、`COMBINED_ENABLE=1`、`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`。

- 逐个开关测试，禁止组合后无法归因。
- 先确认当前版本官方支持和进程继承情况，再跑同口径 30-step。
- 记录普通步、SOAP、CPU 占用、rank 方差、显存和错误；没有稳定收益就撤销。
- 这类结果优先形成启动配置建议，不混入业务算子 commit。

#### R7：`allow_internal_format=True` 实验

- 当前训练入口显式设置为 `False`。这是高风险、全局影响候选。
- 先用 profile 统计 format cast/transpose，证明它们是热点；再在隔离夹具中验证数值与算子兼容。
- 任何不支持算子、精度变化或额外格式转换都应立即终止，不允许为此修改环境版本。

#### R8：动态负载均衡/分桶

- 先统计 8 rank step time、HCCL wait、点数/GT 数/序列长度的相关性。
- 只有 rank 方差和样本复杂度显著相关时，才研究 DrivingSDK load balance 或 dataset bucketing。
- 不得改变 epoch 样本集合、分布式 sampler 契约和随机性；要验证吞吐以及训练分布一致性。

### P3：仅在 profile 命中时研究

- DrivingSDK Patcher：用作隔离实验的非侵入替换工具，避免大面积改客户源码；先研究 patch 生命周期、DDP 多进程导入和可回退性。
- SparseConv/Pillar 全链替换：只有客户当前配置实例化并运行该链路时才做。
- 自定义 rasterizer/几何 kernel：没有成熟等价 API时，最后才考虑自定义 NPU 算子；需要独立正确性规范和更高审核门槛。
- 通信优化/HCCL：只有 profile 显示未掩盖通信或 rank 不均衡成为主要瓶颈时再做；当前不是首批目标。

### 延期项：已注释客户算子的恢复与 DrivingSDK 映射

此项不属于当前 P0～P3 执行队列。另一个模型现在不得进行以下操作：

- 放开或恢复任何已注释 import、调用或自定义算子。
- 为已注释算子插入运行时 hook、构造专项 NPU 测试或捕获恢复后的报错。
- 根据算子名称直接用 DrivingSDK API 替换。
- 把注释算子适配与当前活跃路径优化合并在同一个 patch、实验或 commit 中。

只有用户以后明确要求重新启动这项工作时，才单独建立“可达性盘点 → 一个功能族恢复 → 参考实现等价 → NPU 适配 → 8 NPU A/B”的新计划。届时仍应逐个功能族处理，不能一次性全部恢复。

## 6. 每个候选统一执行模板

1. **可达性门禁**：从有效 config 到 module 实例再到运行时 hook/profile，证明客户负载确实调用。
2. **问题证据**：给出每 step 次数、host/device 时间、busy union/bubble、AICPU 或同步证据；wait-anchor 不能直接算作真实开销。
3. **语义 oracle**：保留旧实现或 PyTorch reference；列出输入、输出、shape、dtype、排序、空输入、动态 shape 和梯度契约。
4. **最小 patch**：一次只改一个机制，保存 diff；先语法和静态检查。
5. **函数级门禁**：真实 NPU forward/backward，对比最大绝对/相对误差、NaN/Inf、dtype/shape。
6. **容量门禁**：8 rank 短跑，确认 HBM、OOM、HCCL、rank 和日志正常。
7. **正式 A/B**：同机器、同卡组、同容器、同 checkpoint、同 config、同 30-step 口径，profiler-off。
8. **裁决**：正确性、普通步、全步、SOAP、吞吐、显存任一关键项明显回归即拒绝；边际收益若小于 2%，建议至少重复 3 次并以 pooled 指标判断。
9. **闭环**：通过则单功能 commit；失败则恢复 HEAD、保留远端诊断证据并写明拒绝原因。
10. **重新 profile**：累计形成显著收益后重采当前 HEAD，重新排序热点，禁止长期沿用旧 profile。

## 7. 统一指标口径

- 30-step 全步均值与全局吞吐：`128 / step_time`。
- 排除 1～3 预热步以及 11/12、21/22 SOAP 周期步后的 23 个普通步 mean/median/P95。
- 两个 SOAP 双步窗及其均值。
- 每 rank framework 峰值显存、`npu-smi` 占用和 OOM。
- 30 个 loss/grad 的有限性、均值、中位数、范围和数量级。
- 8 rank 的完成度、异常日志、HCCL 状态和 step 方差。
- profile 实验另报 device busy union、underfeed、最大内部 bubble、AICPU 暴露、未掩盖通信和 wait-anchor 风险。

## 8. 已否决方向，禁止无新证据重复

- LCFusion 固定 grid 缓存：孤立更快，但正式 30-step 多项轻微回归，已恢复。
- BEVFormer line602 小 tensor：属于异步等待锚点，真实构造开销为亚毫秒级。
- BEV backbone line120：约 18.871 ms/step，低于机制收益门槛。
- MSDA `zeros_like` buffer 删除：理论回收约 2.048 ms/step，低于噪声。
- `torch.isin` 改写 MapTR mask：当前 NPU 微基准更慢。
- NpuFusedAdamW/融合裁剪直接替换 SOAP：接口和更新语义不兼容。
- SOAP QR 强行替换、block 化或升级软件栈：精度/优化器语义风险高，当前 AICPU QR 暂只记录。
- LTO、tcmalloc、驱动/CANN/PyTorch/torch_npu 升级：超出项目授权范围。

## 9. 每个研究项的交付物

- 一页候选报告：问题证据、调用链、预期收益上限、风险。
- 一个最小 patch；或明确的“不修改”结论。
- 一个可复跑的函数级正确性测试。
- 一份 8 NPU A/B 统计，写清命令哈希、HEAD、设备范围和指标。
- 一个单功能 commit，或一条带证据的 rejected 记录。
- 更新 `task_plan.md`、`findings.md`、`progress.md` 和 `操作步骤.md`。

## 10. 可直接发给另一模型的提示词

```text
请严格按照 C:\project\win-project-backup\DongFeng\custom\DrivingSDK优化研究与实施计划.md 继续项目。

先完整读取：
1. C:\project\win-project-backup\DongFeng\AGENTS.md
2. C:\project\win-project-backup\DongFeng\task_plan.md
3. C:\project\win-project-backup\DongFeng\findings.md
4. C:\project\win-project-backup\DongFeng\progress.md
5. C:\project\win-project-backup\DongFeng\操作步骤.md
6. C:\project\win-project-backup\DongFeng\机器IP.md（只用于连接，不得输出凭据）

当前远端权威状态应为 ascend_npu_optimize 分支、HEAD bf9ed6e、Git clean、正确容器 mapqr-leicheng。先只读复核；如不一致，停止修改并报告差异。

第一项只执行 R0：MapTR 两处冗余 .unique() 的两行最小修改。不得顺带修改其他 mask/权重逻辑。依次完成后8卡环境门禁、NPU函数级exact、客户batch16×8 rank×30-step profiler-off A/B。任一正确性或性能门禁失败，恢复且不提交；全部通过才创建单一 commit“【npu性能优化】MapTR正负样本冗余去重消除”，不 push。

R0 闭环后，只从 R1、R2 或 P2 中基于当前 profile 选择一个“当前活跃路径”的研究项。已注释算子的恢复、可达性hook、报错复现和NPU适配全部延期，当前不得执行。任何 DrivingSDK 替换都必须先证明当前活跃调用链可达，再做 forward/backward 等价和正式 A/B；禁止改软件环境。
```

## 11. 官方研究入口

- DrivingSDK README：https://github.com/Ascend/DrivingSDK/blob/master/README.md
- 模型迁移与优化：https://github.com/Ascend/DrivingSDK/blob/master/docs/zh/migration_tuning/model_optimization.md
- Patcher：https://github.com/Ascend/DrivingSDK/blob/master/docs/zh/features/patcher.md
- 数据负载均衡：https://github.com/Ascend/DrivingSDK/blob/master/docs/zh/features/dataload_balance.md
- API 列表：https://github.com/Ascend/DrivingSDK/blob/master/docs/zh/api/README.md
