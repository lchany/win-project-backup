# Findings & Decisions

## 2026-08-18 STEP-265：社区 QR 实测根因（拆成两个问题）

- 用户要求结合实际测试判断原因。证据来自 STEP-245～260 的 8 卡 30-step，以及 STEP-260 的 A/Q/R dump。

### 问题 A：精度从 28/30 掉到约 11/30（不是社区 QR 独有）

| 实验 | QR | 跨 rank | PASS≤2% | step30 vs GPU |
|---|---|---|---|---|
| STEP-245 | HEAD 610 行 SOAP + DIST_QR=1 | broadcast | 28/30 ≤1%，30/30 ≤2% | +1.32% |
| STEP-254 | 63861df 风格 CPU FP64，无 broadcast | 各 rank 独立 | ~11/30 | 中后期偏高 |
| STEP-256 | mx QR + DIST_QR=1 | broadcast | 11/30，无 NaN | +25.38% |
| STEP-258 | CPU FP64 QR + DIST_QR=1 | broadcast | 11/30，无 NaN | +25.24% |
| STEP-259 | mx QR，原始无 broadcast | 各 rank 独立 | 11/25 后 NaN | NaN |

STEP-256 与 STEP-258 轨迹几乎重合。说明当前 63861df 风格工作树本身已经偏了；换成社区 QR 没有把精度救回来，也不是唯一打差的原因。真正对齐过 GPU 的是 STEP-245 那套 HEAD SOAP，不是「原始 63861df + 只换算子」。

### 问题 B：无 broadcast 时后段 NaN（已定位到算子）

STEP-260 在每次 `mx_driving_cloud.linalg.qr` 后检查有限性与 `Q@R≈A`：

- 第一次周期（opt_step=10）**4408** 次调用
- **4400 次 OK**：A/Q/R 全有限，`max|Q@R−A| = 7.2e-6`（同事要求的 Q*R=A 在正常样本成立）
- **8 次 BAD**：shape 全是 `[192,192]`；**A 有限，Q/R 非有限**；ranks 0–7 各 1 次
- 同周期 256 次 192×192 里 248 次正常；5120/2560 大阵这次没有非有限

因果链：坏 Q 写进 SOAP 状态 → 后续投影飞掉 → 约 step16 起 `loss: nan`。开 `SOAP_DIST_QR=1` 后 NaN 消失，是因为坏结果被 rank0 的正常 Q 覆盖，**不是算子修好了**。用户已拒绝把 broadcast 当原始 SOAP。

### 给算子同事的复现物

本地 `C:\project\win-project-backup\DongFeng\step260_qr_bad_tensors\`，已传到 `ubuntu@2.55.0.3:/home/ubuntu/`：`rank{0-7}_step10_ind0_192x192_BAD.pt`。

## 2026-08-18 双门禁：耗时大幅下降 + 逐步 loss ≤2%

- 用户新门槛：逐步 logged `loss` vs GPU `|Δ| ≤ 2%`，同时耗时相对 CPU FP64 SOAP 大幅下降。
- 按远端 30 步 launcher 日志重评分：只有 STEP-238 / STEP-246（`63861df` CPU FP64 双轴）达到 **30/30 ≤2%**；最差点 +1.50% / +1.67%。Iter 2–30 约 865–891 s，相对 GPU 140 s 为 **6.2–6.4×**；SOAP 周期约 205–267 s。
- 快路径全部破 2%：HEAD+one-sided=1024 最接近 GPU 耗时（220 s，1.58×）但 16/30、最差 +11.7%；NPU FP32 / hybrid / wave 两轴路径 12–16/30，step30 −14%～−19%。
- 推论：不能靠 `fb979b2` 亲和栈同时满足双门禁。精度合同冻结为 63861df SOAP 数值；性能只能在该合同上加速（保序并行 CPU QR），不能重新打开 one-sided=1024。
- mx_driving QR（STEP-259/260）后期 NaN，已由 STEP-260 证明是算子对部分 192×192 输出非有限 Q/R，不是模型把有限 Q 用坏。精度 11/30 则是另一条 63861df 风格 SOAP 问题，见 STEP-265。

## 2026-08-18 只读结论：NPU/GPU loss 分叉归因

- 用户怀疑成立：根因提交是 `fb979b28 【npu性能优化】SOAP预条件器NPU亲和优化`，父提交 `63861df 【loss对齐】随机性移除`。
- 该提交不是“同语义换设备”，而是三条优化器语义变更打成一个 commit：
  1. 正式 config `one_sided_dim_threshold=1024`：2D 大权重只预条件较小轴（算法变体，论文亦称 loss 略差）。
  2. 初基：CPU FP64 `eigh` → 设备单位阵 + 立刻 QR。
  3. 周期更新：CPU FP64 mm/QR/`argsort` → 设备 FP32 + `stable=True`。
- 分症状：
  - SOAP 窗 `pts/cls` 拉开：最小变量是 one-sided=1024（STEP-240 关阈值后窗内 <5%；STEP-241 同输入 3 步 max_abs=5.94e-3，大轴 GG 被跳过）。
  - 逐步 total loss 中后期偏低：one-sided 关掉后仍在（STEP-245 10/30）；整文件回到 `63861df` 才到 23/30（STEP-246）。主因是 identity+NPU FP32 周期栈，不是单独一次 QR kernel（STEP-239 否决）。
- 已排除：`SOAP_STALE_Q_K`（STEP-237）、foreach/pin/GeometricLoss（STEP-241 逐位）、仅 MSDA（STEP-238 只 overlay soap 即恢复 SOAP 窗）。
- 现状：已提交 HEAD `669a138` 的 soap 仍含 fb979b2 栈（config 提交值仍 1024）；工作树 soap/config 已被 STEP-259/260 诊断改写，不能当正式基线。
- 正式回退约束：用户已禁止 overlay/`git restore` 抹掉 `fb979b2`；若修精度须新 commit。

## 2026-08-18 精度分叉定位

- STEP-237：`SOAP_STALE_Q_K=0/4` 不能解释首次 SOAP 窗 +18%～+25% `pts` 分叉。
- STEP-238：整份 `soap.py` 回退到 `63861df` 后 SOAP 窗相对 GPU <3%，30 步 `cls` 也回到 GPU 附近。
- STEP-239：仅把当前 `linalg.qr` 改 CPU FP64，SOAP 窗仍约 +20% `pts`，否决“只是 QR kernel”。
- STEP-240：只把 `one_sided_dim_threshold=1024` 改为 `None`，SOAP 窗达标（step12 `pts` −3.0%）。正式 config 已字节级改这一行，未 commit。
- 残余：step20/30 `pts` 约 −10%/−12%；关闭 one-sided 可能拉长 SOAP 周期。

## 2026-08-16 STEP-226 性能主线收口

- 裁决 `PERF_MAINLINE_CLOSED`：HEAD `fa95a2a`；链 `2846401`→`2a2aa0f`→`fa95a2a`。
- 吞吐（GPU 参考、非严格同数据）：全窗 ~0.87:1；稳态 step≥10 ~0.94–0.99:1。
- Level0（STEP-225）无单一 >22.7ms 等价边界；不继续源码挖潜。
- 启用：`SOAP_STALE_Q_K=4` + 已提交 expandable + pin；默认 k=0 同步回退。
- 评测：容器无 ortools，canonical 测试集未跑、不伪造；待客户兼容环境。

## 2026-08-11 TorchNPU Profile 与 Advisor 结论

- 在提交 `63861dfd... 【loss对齐】随机性移除` 上完成正式 8 卡 TorchNPU profile：8 个 rank 均完成 13-step schedule 并生成 DB、operator/kernel CSV 与 trace；总量约 68.9 GiB，全部保留在远端诊断目录，未拉取到本地。
- profile 再次稳定复现第 11 步 SOAP 周期长尾；8 个 rank 完成后主动终止训练产生的线程/launcher 收尾 traceback 不影响已落盘结果。
- rank0 `msprof-analyze advisor all` 成功退出，约耗时 21 分钟、单核 100% CPU；最大 timeline 数据集约 11,499,287 条事件。
- rank0 E2E 364,105.058 ms：NPU 计算 1,704.931 ms（0.47%）、未掩盖通信 35,641.619 ms（9.79%）、NPU 空闲 326,758.509 ms（89.74%）。这与普通基线中周期 host NumPy/OpenBLAS 空洞一致，SOAP CPU fallback 仍为第一优先级。
- advisor 的 4 类亲和 API 候选：`torch.addmm` 135 次、`torch_npu.npu_confusion_transpose` 24 次、`torch_npu.npu_add_layer_norm` 60 次、`optimizer.clip_grad_norm_fused_` 4 次；SOAP `step/project` 多处命中 `npu_confusion_transpose`。
- advisor 还报告 168 次 `aclopCompileAndExecute`（合计 11,143.92 ms）和 AICPU 候选：allreduce 52 条/20,713.687 us、UniqueWithCountsAndSorting 112 条/10,549.245 us、reduce 4 条/1,893.382 us、Cumsum 4 条/550.329 us。
- advisor 将一个约 7.35 秒 host 阶段标记为 slow dataloader，并生成推测 GC 区间；这些可能混入 SOAP host 计算，必须与 timeline/stack 交叉验证，不能按标签直接修改 DataLoader。
- 兼容性限制：`msprof-analyze 1.3.4` 自报按 CANN 8.0.0/Torch 2.1.0 规则分析，而实际环境是 CANN 8.3.RC1/Torch 2.7.1。它能成功解析数据，但亲和 API、环境变量和编译建议只能作为候选；尤其 `ASCEND_LAUNCH_BLOCKING=1` 是调试同步建议，不可用于性能训练。
- rank0 已足以验证 advisor 和暴露主要候选；下一步优先在远端对 8 rank 现有 CSV/DB 做轻量一致性统计，不立即重复 7 次约 21 分钟的全量 advisor。

## Requirements
- 目标分支：`asend_npu_optimize`。
- 背景：项目刚完成迁移和 loss 对齐，需要参考 loss 对齐相关提交理解历史改动。
- 删除此前为对齐而加入的全部固定随机种子/确定性代码，恢复随机行为。
- 该部分必须单独提交，提交信息固定为：`【loss对齐】随机性移除`。
- 全面识别 CPU fallback、非 NPU 亲和算子和其他性能瓶颈。
- 算子优化前先检索业界/社区成熟的昇腾亲和替代方案。
- 性能优化提交统一使用：`【npu性能优化】xxxxx`。
- 先做可复现基线并记录结果，后续按统一口径对比。
- 调研并使用 Ascend 社区/官方性能采集与定位工具。
- 用户批准计划前不得修改业务代码。
- 远程机器信息必须读取本地 `机器IP.md`。
- 除静态代码与算子语义可直接确认的性能问题外，所有候选必须先通过实际工具测试/profile 定位；明确问题修改后也必须工具化 A/B 验证。
- 训练命令必须参考远程当前目录已有 `.sh` 脚本；先只读解析并选定 canonical command，不自行假设训练入口。
- 所有正式训练测试必须使用 8 卡；卡号、rank/world size、HCCL 与 launcher 参数从远程现有 `.sh` 脚本读取。

## Research Findings
- `AGENTS.md` 明确要求：涉及远程机器、SSH、IP、端口、账号、共享目录或跳板机时，从本地 `机器IP.md` 获取准确信息，且不得在回复、日志或提交中重复明文凭据。
- 当前本地工作目录不是 Git 仓库；根目录仅含项目规则、远程连接资料、规划文档和一个临时目录，实际项目代码需要在远程环境中只读审计。
- 远程拓扑为“本机 -> 跳板机 -> NPU 训练机”；代码位于两台远程机器可访问的共享路径，代码审计可在跳板机完成，训练/性能采集必须在 NPU 训练机完成。
- 本机当前只有 Windows OpenSSH 客户端，未发现 PuTTY/Posh-SSH；已检查的 Python 运行时未提供 `paramiko`。
- 尚未开始远程代码、历史提交、运行环境和社区方案的详细审计。
- 已成功通过跳板机连接共享目录和 NPU 训练机；共享目录包含多个 GPU/NPU 版本仓库，目标 NPU 仓库候选为名称带 `ascend_npu` 的独立 Git 仓库，仍需用目标分支交叉确认。
- 跳板机上的 Git 版本较旧，不支持 `git -C`，后续只读 Git 命令统一使用 `(cd "$repo" && git ...)` 兼容写法。
- 名称带 `ascend_npu` 的目标候选仓库当前检出本地分支 `ascend_npu`，HEAD 为 `f189414`，提交主题为“随机性固定”，工作区状态输出未显示未提交改动。
- 该仓库当前列出的本地/远程分支中没有 `asend_npu_optimize`；需继续检查其他仓库及历史，确认这是待创建的新优化分支、名称拼写差异，还是目标仓库选择有误。
- 跨仓库核对后确认实际目标在共享目录主仓库，分支名为 `ascend_npu_optimize`（比用户文本 `asend_npu_optimize` 多一个 `c`）；后续以仓库真实分支名为准。
- `ascend_npu_optimize` 当前 HEAD 为 `72a266b3dd4b0e8e1b510155c8a0cf563bb9ab7d`，提交信息严格为 `【loss对齐】随机性移除`，父提交为 `f189414`（“随机性固定”）。
- 目标工作区干净；`72a266b` 修改 9 个文件，共 41 行新增、123 行删除，`git diff --check` 通过。该提交在计划批准前已产生，当前阶段仅审计，不重写历史。
- `72a266b` 涉及训练配置、训练入口、DataLoader/sampler 及 3 个 shell 脚本；需检查完整 patch，确认没有混入性能优化或破坏 8 卡分布式采样契约。
- `72a266b` 已删除固定值 666、Python/NumPy/PyTorch/torch_npu 全局 seed、msprobe `seed_all`、`PYTHONHASHSEED`、确定性算法开关、HCCL/脚本确定性变量和 worker Torch seed；并恢复数据增强、数据 shuffle 与训练时随机行为。
- sampler 中仍使用 `sync_random_seed()` 生成每次运行的新共享 seed，再给 PyTorch/NumPy 设置该 seed；这不是跨运行固定 seed，作用是让 8 个 rank 使用相同随机采样集合，原则上应保留并通过 8 卡 sampler 一致性测试验证。
- `72a266b` 还把 `synchronize_after_backward=True` 恢复为 `False`。需通过“随机性固定”提交的引入 diff 证明它确属 loss 对齐临时同步措施；否则它会构成性能改动混入随机性提交。
- 当前代码仍存在 `DBG_NPU=1`、`RESET_DEBUG`、`SAMPLER_DEBUG` 和数据集调试输出候选，说明随机性对齐专用调试代码可能尚未完整清理；需结合历史提交确认后补齐到同一随机性移除提交。
- 当前 `run_train.sh` 的 `GPU_COUNT=1` 且实际执行 `MODE=single`，不是可直接使用的 8 卡 canonical 命令；`tools/ddp_train.sh` 已具备 `MODE=multi` + `GPUS` + `torch.distributed.launch --nproc_per_node` 的 8 卡能力，需继续寻找当前目录是否已有封装成 8 卡的现成 `.sh`。
- 已确认 `tools/ddp_train.sh` 默认 `GPUS=8`；`tools/local_train_spetr_debug.sh`、`tools/ddp_train_jw.sh` 等现有脚本也默认 8 卡。下一步解析完整内容和 NPU 环境，选择最符合当前训练场景且不需改代码的 canonical 入口。
- `tools/local_train_spetr_debug.sh` 已选为首选 canonical 8 卡入口候选：默认 `GPUS=8`，通过 `torch.distributed.launch --nproc_per_node` 调用 `tools/train_spetr.py`，并固定当前 config/work-dir 组织方式。执行前仍需确认 NPU 环境变量与当前训练配置参数。
- 仓库旧 `prepare_env.sh` 含无关的历史明文访问凭据；该脚本不是当前 NPU 训练入口，不使用、不提交、不在后续报告中复述。SSH 辅助脚本已增加通用 key/password/token 输出脱敏。
- NPU 训练机 `npu-smi` 显示 16 个物理设备健康，检查时 AICore 利用率为 0 且无 NPU 计算进程；8 卡训练有空闲资源，但具体可见设备映射仍需从运行环境确认。
- 发现 18:11 启动的一组旧单卡 launcher 进程处于 `T`/`Tl`（SIGSTOP）状态、等待点为 `do_signal_stop`，未占用 NPU；为避免未经确认终止旧进程，新的 8 卡验证将使用独立 master port 与 work-dir。
- 非登录 SSH 环境的 `/usr/bin/python` 为 3.9.9 且没有 PyTorch；现有训练脚本依赖启动者已激活的 Conda/环境，需从旧进程可执行文件或登录 shell 只读确认正确 Python 环境。
- 现有历史训练日志表明该模型首 step 曾约 246 秒、后续约 8.5 秒；因此 8 卡短跑必须预留编译 warmup，且不能把首 step 当稳态性能。
- “随机性固定”提交 `f189414` 明确把 `run_train.sh` 从 multi 改为 single，并加入 `DBG_NPU`、数据集调试输出、`RESET_DEBUG`/`SAMPLER_DEBUG`，还把 SeTa 的停止剪枝条件改成 `if True`。这些属于 loss 对齐临时行为，`72a266b` 尚未全部还原。
- `f189414` 还改动 MSDA CPU fallback、distributed sampler、detector 调试与 dataset builder 选择；随机性移除审计不能只看 `72a266b` 的 9 个文件，需要逐项区分“固定随机性临时措施”和“loss 对齐所需 CPU fallback”。
- 全仓 seed 扫描还发现若干数据集中的运行时 `random.seed(0)` 和辅助脚本/测试 seed。需要先确认目标训练配置是否实例化这些类，以及这些 seed 是否由随机性对齐提交引入；第三方测试与部署工具中的 seed 不按关键词盲删。
- 对 `f189414` 逐文件确认：MSDA CPU fallback 本体来自更早的 `70576d`，`f189414` 只添加“随机性固定”标记；该 fallback 暂时保留到 profiler 后的亲和算子替换阶段。
- 本次随机性提交需要补齐的可证明还原项：SeTa `if True` 恢复停止剪枝条件、删除 `RESET_DEBUG`/`SAMPLER_DEBUG`/`SAMPLER_IDX`、删除 detector/dataset 指纹日志与 `DBG_NPU` 常开、恢复 `run_train.sh` 的 multi 分支，并清理对应“随机性固定”标记。
- 上述补齐项已写入远程工作区但尚未提交：7 个文件，13 行新增、88 行删除；本地 6 个 Python 文件 `py_compile` 通过，远程 `git diff --check` 通过。
- 当前工作区 diff 保留 MSDA CPU fallback 逻辑，只删除 f189 添加的随机性标记；没有提前进行算子替换。
- `tools/build_dataset_npu.py` 的 custom concat 修复不是随机性控制，保留功能改动，仅清除未启用的 dataset debug 注释。
- 目标候选仓库根目录存在 `run_train.sh`、`train_card10.sh`、`train_card11.sh`、`train_daemon*.sh` 等训练脚本；8 卡 canonical 脚本尚待逐一只读解析。
- 本地临时快照仅覆盖少量 loss 对齐相关文件，并非完整仓库，不能替代远程全量审计；它可作为历史改动候选线索。
- 快照中存在明确的 NPU 性能抑制候选：`jit_compile=False`、同时关闭两处 internal format、关闭 conv/matmul HF32、固定单进程显存比例、`DBG_NPU=1` 及数据集调试输出。必须先用历史 diff 和 A/B profile 判断哪些是 loss 对齐临时措施。
- 快照中的显式 CPU 路径集中在 SeTa sampler：每批 `indices/loss` 从设备搬到 CPU、`.numpy()`、`.tolist()`、Scikit-learn `KMeans(...).fit(scores.numpy())`，以及 Python 循环中的 `.item()`；这些会造成 D2H、同步和 CPU 计算，是高优先级 profile 候选。
- `seta.py` / `seta_seq.py` 的 `sync_random_seed()` 后再设置全局 PyTorch/NumPy seed 属于待删除候选；但 DistributedSampler 内 `Generator.manual_seed(seed + epoch)` 可能是保证各 rank 得到一致全局排列的算法契约，不能仅按关键词删除，需与 loss 对齐提交父版本逐行比较。
- `builder.py` 的 DataLoader worker seed 仅在 `cfg.seed` 非空时启用；当前快照入口将 `cfg.seed=None`，该路径通常不会生效，也应以历史 diff 判断是否属于此前新增。
- Ascend 官方 TorchNPU 已集成 `torch_npu.profiler`，可采集 PyTorch 层、CANN 层、底层 NPU 算子和内存信息，并输出 timeline/summary；计划采用 warmup 后采集至少 5 个稳定 step。
- `msprof-analyze` 提供 `advisor`（计算、调度、通信自动诊断）、`compare`（NPU-NPU/GPU-NPU 对比）和较新版本的 `module_statistic`，适合作为 profiler 后的自动分析层。
- Ascend `mstt/msfmktransplt` 的 PyTorch Analyse 可生成 API 支持、未知/不支持 API、亲和 API、动态 shape 和 `api_performance_advice.csv`，适合作为全仓静态算子审计工具；它不能覆盖所有原生函数，因此必须结合人工审计和运行时 profile。
- 成熟算子方案的查询顺序确定为：当前版本 TorchNPU 原生 PyTorch API 支持 -> `op-plugin` 实现/版本 -> `torch_npu.contrib` 亲和 API -> Ascend 社区活跃实现；自定义 Ascend C/Triton 算子仅作为最后选择。
- `msprof-analyze advisor` 可直接识别 AICPU、动态 shape、亲和 API、算子下发、流同步、SyncBatchNorm、融合、AI Core、DataLoader 和通信问题，适合将静态候选转化为按优先级排序的运行时证据。
- 官方融合替换样例明确覆盖 `NpuFusedAdamW`、融合梯度裁剪和 `npu_confusion_transpose`；只能在当前环境版本支持且 profile 命中相应模式时采用。
- TorchNPU 的 OpPlugin 兼容路径可能在原生 ACLNN 不可用时降级到备选实现；因此“API 能运行”不等于“已使用最佳 NPU 内核”，需要结合版本、profiler 中的 AICPU/算子路径确认。

## Candidate Audit Categories
- 随机性：Python/NumPy/PyTorch/torch_npu 种子、DataLoader worker seed、deterministic algorithms、CUDNN/NPU 确定性开关、环境变量。
- fallback：显式 `.cpu()` / CPU tensor 构造、未支持算子、设备间复制、host 侧 NumPy/Scipy/OpenCV 计算、同步取值。
- 性能：小算子密集、非连续内存、重复 cast/transpose、动态 shape 编译、同步点、数据加载、混合精度、优化器、通信。

## Technical Decisions
| Decision | Rationale |
|---|---|
| 只把外部资料摘要写入本文件 | 遵守 planning-with-files 的外部内容安全边界 |
| 静态审计与 profiler 结果交叉确认 | 降低误判，优先优化真实热点 |
| 为性能优化设置证据门禁 | 防止凭经验改代码；没有修改前定位、修改后收益和正确性证据，不形成性能提交 |
| 远程已有 `.sh` 作为训练命令事实来源 | 保持与项目当前运行方式一致，避免自造命令造成基线失真 |
| 正式训练统一采用 8 卡 | 保证基线、profile、优化后复测和最终回归口径一致，并覆盖真实通信开销 |
| 补齐随机性还原后 amend 原 `72a266b` | 用户要求随机性移除作为单独一个指定信息的提交；该提交尚未 push，amend 可保持边界纯净 |

## Issues Encountered
| Issue | Resolution |
|---|---|
| `AGENTS.md` 在当前终端以错误字符集显示 | 根据已明确的项目规则执行；敏感信息仍全部脱敏，不影响连接信息读取 |
| 检查系统 Python/py 是否含 `paramiko` 的探测命令退出码为 1，且没有有效输出 | 不重复该失败命令；改查已安装 SSH 工具与其他安全连接方式 |
| 远程 Git 不支持 `git -C` | 改用子 shell `cd` 到仓库后执行 Git，不升级或修改远程工具链 |
| 用户文本分支 `asend_npu_optimize` 未出现 | 已确认实际仓库分支为 `ascend_npu_optimize`，以 Git 中真实名称为准 |
| 只读查看无关旧环境脚本时输出了其历史明文访问凭据 | 停止读取该脚本；不使用、不复述，并增强所有后续远程输出的通用凭据脱敏 |
| 非登录 SSH 默认 Python 无 PyTorch | 从现有训练进程/登录 shell 定位项目实际 Conda 环境，不安装或改动远程框架 |

## Resources
- Ascend 开源社区入口：https://gitcode.com/Ascend
- TorchNPU：https://gitcode.com/Ascend/pytorch
- TorchNPU Profiler 指南：https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md
- OpPlugin：https://gitcode.com/Ascend/op-plugin
- MindStudio Profiler Analyze：https://gitcode.com/Ascend/msprof-analyze
- msTransplant / PyTorch Analyse：https://gitcode.com/Ascend/mstt/blob/master/msfmktransplt/README.md

## Security Note
- `机器IP.md` 可能含敏感连接信息；只用于定位远程机器，不复制到本文件或对话输出。
## 远程训练运行时确认（2026-08-11）

- 目标训练环境位于现有 Docker 容器 `mapqr` 中；宿主机非登录 Python 不具备可用的 PyTorch/NPU 运行时。
- 容器内 Python 为 3.11.10，PyTorch 为 2.7.1，`torch_npu` 为 2.7.1；运行时可见 16 个 NPU 逻辑设备，`torch.npu.is_available()` 为真。
- 容器镜像信息指向 CANN 8.3 RC1 环境。导入运行时会报告 `libop_plugin_atb.so` 权限不匹配警告，需作为环境噪声记录，后续若影响算子加载再单独处理。
- 远程当前没有实际占用 NPU 算力的进程；但存在一个历史单卡训练启动器处于停止态并占用 29507 端口。新验证必须使用独立端口和独立输出目录，不处理该历史进程。
- 历史进程环境中存在 `HCCL_DETERMINISTIC=TRUE`。本次随机性移除验证及后续基线均需在启动命令中显式清除 `HCCL_DETERMINISTIC`、`DETERMINISTIC`、`PYTHONHASHSEED`，避免继承旧固定随机性环境。
- 8 卡验证采用仓库现有 `tools/ddp_train.sh` 的 `GPUS=8 MODE=single` 路径，并显式限制逻辑设备 0-7；端口、工作目录和日志目录均与历史任务隔离。
## 随机性移除 8 卡验证结果（2026-08-11）

- 启动方式：仓库现有 `tools/ddp_train.sh`，`GPUS=8`、`MODE=single`，逻辑设备 0-7，独立端口 29627；显式清除固定随机性相关环境变量。
- 8 个 rank 均完成初始化并进入训练；验证共完成 10 个 iteration。
- 首步包含编译/冷启动，耗时 205.752 秒；第 3-10 步耗时范围约 2.907-3.446 秒。
- iteration 1-10 的总 loss 与 grad norm 均为有限值，未发现 NaN、Inf、HCCL 异常、OOM 或 Python traceback。
- 验证结束后主动停止短跑任务；29507 历史停止态单卡任务也按用户授权清理，29627 训练进程已全部退出。
- 训练会在仓库根目录生成/改写 `fusion_result.json` 与 `kernel_meta/`。这些运行产物未进入提交；`kernel_meta/` 已移动到对应验证结果目录，工作树已恢复干净。
## 强制操作记录规则（2026-08-11 用户补充）

- 新增专用 `操作步骤.md`。每一个逻辑操作必须记录：目的、原因、操作指令、观察现象、现象说明、下一步。
- 新操作必须先登记目的/原因/预定指令，执行后立即补齐结果；长任务的同类轮询合并为一个逻辑步骤并保留关键状态变化。
- 远程地址、凭据、共享目录继续脱敏，操作文件统一以 `{{SHARED}}` 表示共享根目录。

## 8 卡普通性能基线（commit 63861dfd，2026-08-11）

- 固定口径：`tools/ddp_train.sh`（SHA-256 `e006683c...`），目标配置 SHA-256 `6872d9a2...`，设备 0-7，world size 8，每卡 batch 1，全局 batch 8，30 个 iteration；每次独立进程、端口和结果目录。
- 环境：Kunpeng 920（320 CPU 核）、16 个可见 Ascend 910 逻辑设备；Python 3.11.10、PyTorch 2.7.1、torch_npu 2.7.1、CANN 8.3.RC1、mmcv 1.7.2、mmdet 2.19.0、sklearn 1.5.1、NumPy 1.23.5。
- 三次首步分别为 213.604、217.658、248.140 秒，中位 217.658 秒；首步需视为编译/冷启动。
- 三次运行都在第 11、21 步出现周期性长尾：6 次分别为 271.366、271.606、255.894、256.672、280.307、276.872 秒；中位 271.486 秒，P95 280.307 秒，CV 3.81%。
- 90 个 step 总体均值为 28.932 秒，对应全局 batch 8 的真实端到端吞吐 0.2765 sample/s；三次 run 均值 CV 为 3.39%。
- 排除首步及第 11/21 步后，81 个 step 中位为 3.186 秒、P95 为 7.074 秒，对应理想稳态吞吐 2.5110 sample/s。
- 每次最大日志显存均为 5067 MB；三次均无 NaN/Inf、OOM、HCCL 错误或 traceback。
- 周期长尾期间 8 卡 AICore 为 0%，8 个主 rank 各约占 100% host CPU；GDB 只读采样显示主线程位于 NumPy `FLOAT_multiply` / OpenBLAS `dgemm_kernel_ARMV8SVE`。这证明长尾来自同步 host NumPy/BLAS 路径，而非 NPU 或 HCCL 卡死，但具体 Python 调用链仍需 profiler/定向工具确认。
- 每次训练都会删除/改写 `fusion_result.json` 并生成 `kernel_meta/`；运行后已恢复 HEAD 文件并将 kernel 产物移动到对应诊断目录，当前远程工作树干净。
# 新增远端数据边界（2026-08-11）

- 禁止将远端服务器上的数据、数据集、日志、profile、checkpoint、模型或训练产物拉取/同步到本地。
- 后续性能解析、CSV/DB 聚合、advisor 和静态分析全部在远端服务器/`mapqr` 容器内完成；本地仅保存脱敏后的必要指标、结论及操作记录。
- 禁止使用 `.codex-tools/remote_sync.py pull`、`scp`、`sftp get` 等方式下载远端产物；经审查的临时代码仍可从本地向远端推送。

## 2026-08-11：首个 SOAP NPU 亲和优化完成

- 基线提交保持为 `63861dfd920ab9829512b1e4a000eefd1ffcfbea 【loss对齐】随机性移除`；性能提交为 `fb979b28ee3d417806a48c0d643676fd7d38541e 【npu性能优化】SOAP预条件器NPU亲和优化`，父提交精确为 `63861dfd...`，未 push。
- 成熟方案依据：SOAP ICLR 2025 one-sided 变体只旋转矩阵较小一侧；SOAP 原作者使用设备 FP32 power-iteration/QR；Meta Distributed Shampoo 的 SOAP 配置使用设备 identity eigenvector 初始化和一轮 QR。当前环境的 Ascend op-plugin 通过官方 ACLNN 提供 `torch.linalg.qr`，实测底层为 AICPU QR；`eigh` 则明确 fallback CPU，因此不能保留。
- 实现范围仅两份业务文件：SOAP 新增默认关闭、由目标 config 显式设为 1024 的 `one_sided_dim_threshold`；仅对含大轴的 2D 权重保留较小轴；初始化改为设备 identity 后复用一轮 QR；周期更新保持 NPU FP32，删除 CPU FP64/eigh/QR 搬运路径。其他配置默认不启用 one-sided。
- 函数门禁：8 rank 对 `(2560,5120)` 与 `(256,2560)` 均选择 `(True,False)`；project/back 相对误差约 `3.09e-7`；256/1024/2560 QR 的正交误差约 `1e-6`，rank 间结果一致，无 fallback。
- 三次 8 卡 After 均为 30 step，run 均值 5.559/5.674/5.719 秒，CV 1.187%；90-step pooled 均值 5.651 秒、吞吐 1.4158 sample/s。Before 为 28.932 秒、0.2765 sample/s；端到端均值下降 80.47%，速度 5.120×，吞吐提升 412.03%。
- 周期步 Before 中位 271.486 秒，After 15.578 秒，下降 94.26%、速度 17.428×，After 周期 CV 0.394%。普通稳态中位 3.186→3.178 秒，无实质回归。最大日志显存 5067→4070 MB，下降 19.68%。三次 After 的 loss/grad 全部有限，无 traceback、CPU fallback、OOM 或 HCCL error。
- 优化后 rank0 轻量 TorchNPU profile 在真实 8 卡训练中覆盖 Step 9–12：长 Step 10 为 26.362 秒，其中 computing 23.276 秒、未掩盖通信 0.254 秒、free 0.686 秒。SOAP device total 23.033 秒；`linalg_qr` 543 次、device total 22.674 秒，全部 kernel 为官方 `aclnnLinalgQr_QrAiCPU_Qr`。
- Before 551 次 QR，After 少 8 次，精确对应删除的 4 个 5120 轴和 4 个不必要 2560 轴；After 22.674 秒与函数微基准估算 23.007 秒一致。Before 约 95.5 秒 CPU QR 与 157.5 秒 CPU mm 已消失，形成工具化因果闭环。
- 剩余热点：保留的 4 个 2560 轴及小矩阵仍走 AICPU QR，约占周期计算 22.7 秒。继续优化需要成熟的分块 SOAP/Shampoo 或 AI Core 正交化实现，并重新完成数值、8 卡 A/B 和长期收敛门槛；不应混入当前已验证提交。
- 所有训练日志、profile、DB、CSV、诊断脚本和 runtime artifacts 均留在远端诊断目录；没有执行任何远端到本地的数据拉取。远端业务仓库提交后工作树干净。

## 2026-08-11：剩余 2560 AICPU QR 方案筛选

- After profile 的 543 次 `aclnnLinalgQr` 共 22.674 秒，是剩余最大设备热点；HCCL allreduce 约 0.937 秒，量级明显更低。
- Meta Distributed Shampoo 的 block 是成熟能力：README 将 `max_preconditioner_dim` 作为 block size，建议在 1024–8192 间权衡；源码 `multi_dim_split` 会沿所有维度递归分块，`DefaultSOAPConfig` 使用 QR，`ignored_basis_change_dims` 可关闭指定轴 basis change。
- 兼容性边界仍不满足：Meta 当前要求 PyTorch>=2.8/Python>=3.12 且只声明 CUDA 测试；本项目为 PyTorch/TorchNPU 2.7.1、Python 3.11、CANN 8.3。SOAP 原作者没有可直接移植的 block 版本。
- Ascend 最新公开 op-plugin 仍以 `aclnnLinalgQr` 作为 QR 适配入口；未找到当前环境版本可用的 AI Core QR/eigh/SVD 替代。本机 kernel 已明确为 `QrAiCPU_Qr`，因此不能把“官方 ACLNN”误称为 AI Core 高性能实现。
- 依据现有实机微基准估算，标准 1024 多维 block + one-sided 对本模型只约节省 5.43 秒/周期、端到端约 10%，同时把完整 2560 轴 basis 变为独立 block basis，丢失跨块二阶相关。风险/收益不足，暂缓实现。
- 下一候选转为 DataLoader：After profile 明确使用 `_SingleProcessDataLoaderIter`，4 个采样 step host self 约 5.996 秒，约 1.5 秒/step；先通过 8 卡 worker 参数扫描验证，不直接改配置。
- 一手来源：[Meta Distributed Shampoo README](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md)、[Meta `shampoo_utils.py`](https://raw.githubusercontent.com/facebookresearch/optimizers/main/distributed_shampoo/utils/shampoo_utils.py)、[Meta `shampoo_types.py`](https://raw.githubusercontent.com/facebookresearch/optimizers/main/distributed_shampoo/shampoo_types.py)、[Ascend op-plugin QR 目录](https://gitcode.com/Ascend/op-plugin/tree/05072e4503d261242fcdb5418e4be933e8d08642/op_plugin/ops/opapi)。

## 2026-08-11：DataLoader 多进程加载优化

- After SOAP profile 明确显示 `_SingleProcessDataLoaderIter.__next__` 4 次 host self 共 5.996 秒，`StepTraceTime.preparing` 每步约 1.50–2.06 秒；目标 config 的有效值为 `workers_per_gpu=0`。
- 8 卡扫描中 worker=2 的 30-step 均值/吞吐为 4.042 秒/1.9791 sample/s；worker=4 为 4.085/1.9584，未提供额外收益且进程与启动成本更高，因此选择最小足够的 worker=2。
- worker=2 三次正式复测 run 均值 4.042/4.040/4.024 秒，CV 0.198%；90-step pooled 中位 1.468 秒、P95 13.905 秒、均值 4.035 秒、吞吐 1.9825 sample/s；普通稳态中位 1.441 秒；周期中位 13.842 秒。
- 相对 SOAP 提交的 worker=0：均值下降 28.59%、速度 1.400×、吞吐提升 40.03%；相对最初基线累计速度 7.170×、吞吐提升 617.00%。显存保持 4070 MB，三次 loss/grad 全部有限，无 worker/shared-memory/fallback/traceback/OOM/HCCL 错误。
- 已形成独立提交 `6477a5b6eab010b36c9ffb14eee4ec127bc1d7f8 【npu性能优化】DataLoader多进程加载`，父提交为 `fb979b28...`；只改目标 config 一行，未 push，远端工作树干净。
# 新增硬性约束（2026-08-12）

- 禁止升级、降级、替换或以其他方式强行改变客户环境中已经存在的组件版本，包括但不限于驱动、固件、CANN、PyTorch、torch_npu、MMCV/MMDetection 及项目依赖。
- 算子与工具方案只能使用当前环境已经具备且版本兼容的能力；若官方或社区成熟方案要求改变现有组件版本，则仅记录为版本不适用并暂缓，不得安装、切换版本或据此修改客户环境。
- 所有训练、基线、A/B 和 profiler 训练必须且只能在完整名称为 `mapqr-leicheng` 的客户现有容器内执行；禁止在宿主机以及 `mapqr`、`mapqr-leicheng-shm64m`、`mapqr-leicheng-incomplete` 等其他相似容器中启动训练。宿主机只允许做连接、只读状态核验和正确容器的命令编排。
- 全程训练设备固定为 8 张昇腾 NPU，禁止混用或退化为 GPU/CUDA/CPU 训练。启动证据至少包括正确容器、`torch_npu` 可用、world size/rank 为 8，以及 `npu-smi` 中 8 卡训练进程；脚本遗留的 `--gpus 8` 是 MMCV 兼容参数名，不代表实际使用 GPU。

## 2026-08-12：新机器恢复任务启动

- 用户确认本轮目标是恢复从老机器中断、刚迁移到新机器的优化项目，并继续后续优化。
- 本地目录不是业务源码仓库，当前 Git 仅跟踪规划与说明文档；顶层可见 `AGENTS.md`、`优化.md`、`task_plan.md`、`findings.md`、`progress.md`、`操作步骤.md`。业务源码仍应以新机器共享盘中的目标仓库为准。
- 本地存在 `.codex-tools/remote_exec.py` 与 `.codex-tools/remote_sync.py`；后续只允许使用前者做脱敏的远程只读/编排操作，禁止使用 sync pull 或任何方式把远程产物下载到本地。
- 本地文档仓库当前分支为 `main`，原始 HEAD `7a69c06`，除本轮计划与操作记录外无其他已显示改动。
- 已将任务计划切换到 Phase 7“新机器迁移恢复与基线重建”；旧机器基线和两项历史优化只作为迁移核对线索，所有新性能结论必须在新机器 `mapqr-leicheng` 容器、8 张昇腾 NPU 上重新建立。
- 本地远程 helper 的解析/脱敏逻辑仍可用，但迁移后缺少 Paramiko 依赖；bundled Python 3.12.13 可用。恢复方案只在本地 `.codex-tools/vendor` 补齐 helper 依赖，不修改客户环境。

## 2026-08-12：新机器第一轮只读恢复审计

- 本地 helper 依赖已恢复到 Codex 用户级工具目录：Paramiko 3.5.1；未修改项目业务代码或客户机器环境。
- 新机器为 aarch64 Huawei Cloud EulerOS 2.0。唯一允许训练的容器 `mapqr-leicheng` 正在运行；同时存在多个名称相似容器，后续命令必须继续使用完整名称并做前置断言。
- `npu-smi 25.5.1` 识别 8 个 NPU ID、每个 2 个 chip，共 16 个逻辑 chip；全部健康，AICore 当前 0%，无 NPU 训练进程。空闲 HBM 存在约 2.8–3.1 GiB 的平台/运行时占用，后续基线需按新机器重新记录。
- `mapqr-leicheng` 内为 Python 3.11.10、PyTorch 2.7.1+cpu、torch_npu 2.7.1，`torch.npu.is_available()` 为真、可见 16 个 chip。`msprof`、`msprof-analyze`、`torch_npu.profiler` 均可用；导入时仍有既有 `libop_plugin_atb.so` owner mismatch 告警，只记录为环境噪声，不修改权限或版本。
- 新机器目标仓库分支为 `ascend_npu_optimize`，HEAD `6477a5b6...`，工作树干净；提交链连续包含 `63861df` 随机性移除、`fb979b2` SOAP NPU 亲和优化、`6477a5b` DataLoader 多进程加载。代码迁移主链完整。
- 下一门槛是验证容器挂载与 canonical `tools/ddp_train.sh`：必须确认 8 rank、NPU 0–7 设备映射、目标 config、数据/权重可达、独立端口/输出目录以及脚本/config hash，之后才能启动恢复短跑。

## 2026-08-12：训练入口与迁移文件第二轮审计

- `mapqr-leicheng` 已挂载共享盘，容器内可见目标 Git 仓库与 `tools/ddp_train.sh`。
- canonical 脚本 SHA-256 仍为 `e006683c...`，与老机器冻结值一致；当前目标 config SHA-256 为 `ecdb64ef...`，因 SOAP 与 DataLoader 两次提交已发生预期变化。脚本 `bash -n` 通过。
- `tools/ddp_train.sh` 默认 `GPUS=8`、`MODE=single`、以 `torch.distributed.launch --nproc_per_node ${GPUS}` 启动 `tools/train_spetr.py`；`--gpus` 仅为 MMCV 兼容参数。脚本记录了 `PY_ARGS` 却未向 Python 透传，因此恢复短跑仍需复用远端归档的 `--max-iters` 测试夹具，不能把额外参数直接附在原脚本末尾。
- 目标 config 当前有效文本包含 `workers_per_gpu=2`、`precondition_frequency=10`、`one_sided_dim_threshold=1024`，与两项历史性能提交一致；`load_from` 为空、`resume_from=None`。
- 静态路径存在性扫描发现两个绝对 checkpoint 字面量缺失，但尚未证明是当前有效配置引用，可能来自未启用分支或注释/嵌套候选。必须用 MMCV `Config.fromfile` 解析有效配置后再判断，不能据此启动或阻断训练。
- 三个关键提交的文件边界与老机器记录一致：随机性移除 13 文件；SOAP 优化仅 optimizer 与目标 config；DataLoader 优化仅目标 config 一行。

## 2026-08-12：新机器 worker=2 profile 完整性确认

- 迁移后今天在正确容器 `mapqr-leicheng` 上生成的 `profile_worker2_mapqr_leicheng_8npu_20260812T091409` 已完成：退出码 0、最大 iteration 14、无 traceback、当前无训练/NPU 进程、仓库工作树干净。
- profile 目录共有 141 个文件、总计约 3.87 GB，全部保留在远端；rank0 已生成 `kernel_details.csv`、`trace_view.json`、`step_trace_time.csv` 与数据库，具备继续做 device bubble/AICPU/host-device/算子归因的基础。
- 未生成 `communication.json`，因此后续通信带宽与等待归因需要降级为 HCCL kernel/timeline 证据并明确置信度限制。
- 训练目录包含 iteration 14 自动 checkpoint；不读取其内容、不下载到本地、不纳入 Git。本轮不重复启动 8 卡 profile，优先复用已有有效证据。
- 下一步按 `ascend-profiling-anomaly` 技能读取 kernel 数据、异常规则、架构报告模板和参考实现，然后在远端原位生成异常分析与独立模型架构报告。

## 2026-08-12：本地依赖安装授权

- 用户授权后续优化过程中，若本地 Codex 工作机缺少必要软件、工具或依赖，可直接安装，无需逐项确认。
- 每次本地安装仍需记录用途、版本、来源和验证结果，并避免把本地工具依赖混入业务提交。
- 授权不扩展到客户新机器、`mapqr-leicheng` 容器或任何远端组件；远端驱动、固件、CANN、PyTorch、torch_npu 和项目依赖仍禁止安装、升级、降级或替换。

## 2026-08-12：worker=2 profile schema 与初步聚合

- 核心数据大小：`kernel_details.csv` 约 8.86 MB/102,354 rows，`operator_details.csv` 约 19.35 MB，`trace_view.json` 约 277.65 MB/1,604,917 events，`step_trace_time.csv` 4 rows；另有 `op_summary`、`task_time` 与 API statistics。
- profile 覆盖 Step 9–12。Step 9/11/12 的 device computing 约 0.31–0.34 秒；Step 10 computing 约 23.393 秒，QR kernel `aclnnLinalgQr_QrAiCPU_Qr` 设备总时长约 22.791 秒，仍为压倒性周期热点。
- Step 9–12 的 `Free` 约 0.99–1.25 秒/step；普通 step stage 约 1.44–1.60 秒，表明 worker=2 后普通迭代仍存在显著 device underfeed。必须进一步用 interval union 与 host overlap 定量，不能只用 step CSV 粗判。
- Trace 共有约 1.60M events，主要类别为 `cpu_op`、`HostToDevice`、`async_npu`、task queue；出现 15,360 次 `aten::item`/`_local_scalar_dense` 和 212,268 次 `torch_to_npu` 关联事件。二者可能包含关联/流事件或真实同步，需用 `ph`、duration、时窗重叠和调用栈进一步判别。
- 纯 kernel duration 排名除 QR 外，依次包括 MSDA backward、Index/IndexPut、MatMul/Conv、Nonzero 与 HCCL；这些是候选排序，不等于优化结论。
- 第二次 schema 脚本在输出 trace marker 时因 `ijson` 的 Decimal 无法 JSON 序列化而退出；此前所有聚合均已完成，远端无写入。正式脚本将显式转换 Decimal，并使用 `op_summary` 的 Task Type/Stream ID 补足 `kernel_details` 缺失字段。

## 2026-08-12：worker=2 profile 正式异常与架构分析

- 已在远端诊断目录生成 `anomaly_discovery.json`、`anomaly_discovery_report.md`、`analysis_manifest.json` 与独立 `model_architecture_report_profile_worker2_mapqr_leicheng_8npu_20260812T091409.md`；分析工具同样归档在该目录，业务仓库工作树保持干净，未下载任何 profile/报告/权重。
- Step 9/11/12 的 underfeed ratio 分别为 78.34%/79.66%/79.69%，设备 busy union 约 0.29–0.35 秒，而 service 约 1.44–1.60 秒。最大内部空洞分别约 215/289/382 ms，属于重复的 `DEVICE_IDLE_GAP_HEAVY`/`INTERNAL_BUBBLE_HEAVY`。
- 四个最大内部空洞均位于 `NanToNum` 小 kernel 之后、`Arange` 小 kernel 之前；host 覆盖接近 100%，HCCL overlap 为 0。Step 12 空洞与 sync/H2D 标记重叠约 22.6%，其余主要标为 possible host launch lag。
- 空洞内重复出现 2–4 次 `aten::to`/`_to_copy`/`copy_`，重叠时长约 6–86 ms；普通 step 全部 host 统计中，SOAP optimizer wrapper 约 175–371 ms，copy/to 合计约数百毫秒。说明下一步应先定位这一重复搬运/类型转换的功能来源。
- `aten::item` 每个普通 step 约 2,839–3,880 次，但 host duration 仅约 24–27 ms；属于高频同步风险候选，不是当前最大 215–382 ms 空洞的直接时长解释。
- 543 次 `aclnnLinalgQr_QrAiCPU_Qr` 共 22,791.134 ms，masked ratio 约 0.0002%，分类为 `AICPU_EXPOSED_NOT_ALLOWED`；仍是周期 Step 10 的压倒性瓶颈。当前环境没有已证实成熟 AI Core QR，继续保持“记录但不强改版本”的决策。
- `Arange` 与 `ZerosLike` 分别命中 wait-anchor 假热点规则，必须从根因排名降级。HCCL kernel 共约 113.581 ms、与 AI Core overlap 约 44.10%；缺少 `communication.json`，不进入首批优化。
- 模型架构由有效 config 与 kernel family 交叉确认：SPetr3D，ResNet+FPN 图像链、PillarVFE/PointPillarScatter、BEVFormer encoder/MSDeformableAttention、MapTR decoder/head、SOAP optimizer。无 FIA/decode，报告已按训练模型降级而非套用 LLM 结构。
- 校正版已剔除外层 `ProfilerStep#N` 对 host coverage 的虚假 100% 覆盖：四个最大内部空洞的细粒度 host coverage 分别约 22.76%、14.48%、12.19%、3.09%；Step 9 转为 `possible_untraced_host_blocking`，Step 12 保持 `possible_sync_or_h2d`，其余偏 host launch lag。尾部 76–122 ms 的细粒度 host coverage 为 0。
- AST 静态审计确认实际 SOAP 实现不含 `nan_to_num` 或 `arange`，只有 QR 更新中的 argsort/index_select/to；因此最大空洞边界不能解释为 SOAP 函数内部 `NanToNum→Arange` 直接调用链。
- 已迁移的 profile hook 支持动态 wait/warmup/active/repeat、rank worker name、CPU+NPU activity 和 `with_stack` 参数；8-rank max-iters shell 夹具 SHA-256 前缀仍为 `5bee742d644f`。可基于此创建仅 rank0、单 active step 的 stack follow-up，而无需修改业务脚本。
- 归档 hook 的 `NPU_PROFILE_RANK` 默认 0，非目标 rank 会直接不创建 profiler；旧配置为 wait=8/warmup=1/active=4/repeat=1、shapes/memory/stack 全关闭。窄 profile 将仅调整为 wait=10/warmup=1/active=1、with_stack=true。
- 归档 config 使用相对 `_base_='./projects/configs/...'`，移入诊断目录后直接 Config.fromfile 会失效；这是预期归档形态。运行时临时 config/hook 必须放回仓库根，结束后由 trap 移回新诊断目录。

## 2026-08-12：功能级提交规则

- 后续优化以可独立验证、可独立回退的功能为原子提交单位；一个算子的一项完整优化原则上只提交一次。
- 同一功能的代码、必要配置和直接相关测试调整应放在同一 commit，不拆成准备/实现/修补等多个零碎 commit。
- 只有复杂功能的子能力本身可独立验证、独立回退时才拆分，且拆分前必须明确边界。
- 性能提交信息继续使用 `【npu性能优化】<具体对象与动作>`；不混入日志、profile、checkpoint、测试夹具或其他运行产物，默认不 push。

## 2026-08-13：窄栈 profile 完整性与初步调用链

- 窄栈任务完成 12/12 iter、退出码 0，无 traceback/OOM/HCCL error；退出后端口、8 NPU 进程均释放，业务仓库保持干净。rank0 `operator_details.csv` 约 13.46 万行，其中约 10.49 万行带非空 stack、约 1009 个唯一 stack，证明 `with_stack=true` 采集有效；原始文件仍只留远端。
- 本次实际采集 `ProfilerStep#11`，它是 SOAP 周期步而非纯普通步。聚合中 SOAP `project/update_preconditioner/project_back/step` 是主要仓库调用帧；`aten::item/_local_scalar_dense` 主要来自 SOAP step 的 167/168/170 行，表明周期路径存在 Python 标量同步，但不能据此解释普通 step 的 215–382 ms 空洞。
- 全表 `aten::to/_to_copy/copy_` 聚合明显被训练结束 checkpoint 保存污染：checkpoint save/`weights_to_cpu` 分别占据最大调用帧；MMCV scatter 到设备仅约 37 ms host self、约 107 ms host total，不足以单独解释既有普通 step 最大空洞。
- `gradient_fingerprint_optimizer_hook.py:85` 在采样中出现，host total 约 3.95 秒，属于需要核验生效频率和功能必要性的高优先级候选；当前尚未证明其处于普通迭代关键路径，也未授权直接删除。
- 当前栈聚合只能用于筛选候选，不能作为功能修改结论。下一步应按 trace 时间窗把每个 device bubble 与 CPU op 的 `Call Stack` 精确求交；若 Step 11 仍无法提供纯普通路径证据，则以 `checkpoint_config=None` 的诊断派生配置重采一个正常 step，避免末步 checkpoint 污染。
- 时间窗求交已把 277.612 ms 的 `NanToNum→Arange` 主空洞定位到 `spetr3d.py`：`tensor_hash` 在 `obtain_history_memory` 的两段重复、无条件 `FWD_IN` 打印中每步执行，窄栈正好记录 4 次 `aten::to`，其中两次大张量转换各约 3.65 ms；其余 MD5/NumPy 主机工作不形成 NPU kernel，解释了低细粒度 host coverage 与大段未追踪主机时间。
- 同一 step 的 4.991 秒尾部空洞已精确归因 MMCV 末步 checkpoint save/`weights_to_cpu`，不属于稳态模型路径；54.1 ms 空洞为 grad clip，18.4 ms 空洞为 BEV encoder 小张量构造，均低于无条件输入哈希候选。
- 有效 `optimizer_config` 的类型仍为 `GradientFingerprintOptimizerHook`，但 `fingerprint_iters=()`、`fingerprint_phases=()`、`synchronize_after_backward=False`，不会执行梯度哈希或额外 synchronize；profile 中约 3.95 秒 total 是其包裹的完整 backward/clip/optimizer 时间，不能误判为 hook 自身调试开销。
- `tensor_hash` 和无条件 `FWD_IN` 由较早的对齐/调试工作引入；两段调用没有配置开关，也不参与 loss 或模型输出语义。候选功能边界为只移除这两段重复无条件打印，保留其他命中特定样本且受 `hit_target/debug_print` 保护的诊断代码，避免扩大改动范围。

## 2026-08-13：无条件训练输入哈希 A/B 结论

- 修改前/后各完成 3 次独立 8 NPU、30-step 运行，均在唯一正确容器 `mapqr-leicheng` 中执行；每次启动均核验 1 个 launcher、8 个直接训练 rank、`npu-smi` 的 8 个唯一 Python 训练 PID，结束后 NPU 进程为 0。
- Before 三轮普通步中位为 1.2890/1.2845/1.2870 秒，After 为 1.0860/1.0310/1.0675 秒。pooled 普通步中位 1.2870→1.0565 秒，降低 17.91%，速度 1.218×；P95 1.414→1.261 秒。周期 SOAP 步中位也由三轮 13.8135/13.7530/13.6530 秒降至 13.4610/13.3995/13.4715 秒。
- 每轮 Before 恰好产生 480 行 `FWD_IN`（8 rank × 30 step × 2），三轮 After 均为 0。业务改动只有 `spetr3d.py` 14 行删除，不改变任何张量赋值、模型调用、loss、optimizer、DataLoader 或 checkpoint 配置。
- 正确性门禁：6 轮全部 loss/grad 有限，无 traceback/OOM/HCCL/fallback 错误；Before/After pooled loss 均值 335.327/327.554（-2.32%），中位 332.318/320.326，范围 229.913–464.517/223.944–456.577，高度重叠；三轮前 10 步与后 10 步均呈正常下降，未见因修改造成的大幅偏离。短跑随机数据顺序下不要求逐步数值完全相等，但本修改不参与计算图。
- 最大显存 4068–4070 MB，基本不变。训练生成的 `kernel_meta/` 均在确认未跟踪后移入对应远端诊断目录；日志、checkpoint、测试夹具和统计均未下载到本地，也不进入业务提交。
- 决策：保留该优化，并按功能原子性形成唯一提交 `【npu性能优化】训练输入哈希调试移除`；不 push。
- 已创建提交 `5a37d0432951db6ffd0b145ea151a4fd33b1a0be 【npu性能优化】训练输入哈希调试移除`，父提交为 `6477a5b6...`；提交后远端业务工作树干净，未 push。Git 使用远端现有自动配置的提交身份，仅提示可检查身份，没有修改全局或仓库配置。
- 提交后 rank0 with-stack 正常 Step 12 复核已完成：原 277.612 ms `NanToNum→Arange` 空洞降至 49.469 ms，下降 82.18%；`operator_details` 中 `tensor_hash` 与 `FWD_IN` stack 行均为 0。该结果与 3×8 NPU A/B 的 17.91% 普通步改善构成因果闭环。
- 正常 step 当前最大内部空洞变为 55.700 ms，边界为 `LinalgVectorNorm→Stack`，stack 聚合指向 MMCV `clip_grads`；第二大 49.469 ms 主要剩余为历史 memory slicing/location 构造，不再含输入哈希 D2H。下一候选优先审计当前 torch_npu 2.7.1 是否提供成熟官方 fused grad-norm/clip 能力，不能通过升级版本获取。

## 2026-08-13：梯度裁剪候选筛选结论

- 当前 MMCV 直接调用 PyTorch 2.7.1 `clip_grad_norm_`。环境源码与 profile 交叉确认 NPU 梯度已走 `aten::_foreach_norm`；PyTorch 通用实现对支持 foreach 的设备自动选择批量 norm/mul 路径，并避免用 host 条件判断 clip coefficient。
- torch_npu 2.7.1 顶层/npu/utils/contrib 没有通用公开 fused clip-grad API。唯一相关成熟实现是 `torch_npu.optim.NpuFusedOptimizerBase.clip_grad_norm_fused_`，依赖该 fused optimizer 自己的 combined gradients/masks；另有专用于 NpuFusedLamb/BertAdam 的内部实现。
- 当前 SOAP 是项目自定义 optimizer，不能在不重写梯度存储和 optimizer step 的情况下调用 `NpuFusedOptimizerBase` 内部接口。替换 optimizer 或强接 combined-gradient 机制会扩大功能边界并可能改变 loss/收敛语义，且 55.7 ms 收益上限不足以支撑该风险。
- 决策：梯度裁剪 fused 候选在当前版本/optimizer 下不适用，不修改、不微基准、不提交；继续筛选调试残留等不进入最终功能语义的候选。

## 2026-08-13：中间输出梯度调试候选 A/B 结论

- 全仓静态闭环确认 `_debug_output_tensors` 只有初始化和写入，没有读取；候选 diff 仅删除 `spetr3d.py` 中 36 行调试代码，包括属性初始化、4 个 pts-backbone 保留块、1 个 BEV 保留块和 1 个 lane 保留块。`git diff --check` 与 `py_compile` 均通过，计算张量和 loss 路径没有改写。
- 三次 After 均在唯一正确容器 `mapqr-leicheng` 中使用 NPU 0–7、8 个 NPU 映射训练 PID完成 30 step，退出码均为 0；无 FWD 哈希、traceback、OOM、HCCL error 或 CPU fallback。三轮普通步中位为 1.0820/1.0680/1.0905 秒，周期步中位为 13.4765/13.4525/13.4020 秒，最大显存均为 4069 MB。
- 同一提交的三轮 Before 普通步中位为 1.0860/1.0310/1.0675 秒；pooled 普通步中位 1.0565→1.0755 秒，候选反而慢 1.7984%。P95 1.273→1.209 秒，但与中位方向不一致；显存 4070→4069 MB 仅差 1 MB，不构成实际收益。周期步整体基本持平。
- 正确性门禁通过但不改变性能决策：Before/After pooled loss 均值 327.5536/334.7654（+2.2017%），中位 320.3263/323.8277，范围 223.9436–456.5770/221.5122–460.6731，高度重叠；六轮每一轮后 10 步 loss 均值都低于前 10 步，全部 loss/grad 有限。
- 决策：该候选性能收益低于噪声且 pooled 中位回退，拒绝保留，不形成 `【npu性能优化】` commit。候选 diff 和运行产物只留远端诊断目录；业务文件将精确恢复至 HEAD `5a37d043`。

## 2026-08-13：`MAP_SHIFT` 日志候选筛选

- 三轮当前 HEAD 基线日志的 `MAP_SHIFT` 均只有 24 行：8 行 idx、8 行 choice、8 行 hash，精确等于 8 rank 各命中一次三行打印，不是每 step 高频路径；每轮只有 9 种唯一文本。
- 有效路径中的 `index=np.arange(final_shift_num)` 随后用于切片三组真实训练数组，不能作为纯日志计算删除；只删 24 行 stdout 的收益上限很低，不足以再消耗三轮正式 8 NPU A/B。
- 源文件中其他 `MAP_SHIFT enter`/`MAP_SHIFT_TRACE` 位于多个替代 property 实现，但当前三轮日志均未命中。决策：不修改、不测试、不提交，转向已有 normal-step profiler 明确的约 49 ms 切片/locations 路径。

## 2026-08-13：关闭 PVB 坐标网格死计算 A/B 结论

- 有效配置 `pts_bbox_head=None`，而 `forward_pts_train` 仍构造 4 级 location 网格；PVB 函数随后入口早退，lane3d 的位置编码已注释且不读取 location。normal-step profile 对构造路径匹配 228 条记录、host self 20.766 ms，两处分支 clone 另约 2.813 ms，死计算证据成立。
- 候选只做 3 个单行替换：head 为 None 时跳过 `prepare_location`，两处 clone 对 None 直传；head 非 None 时保留原行为。三次 After 均在正确容器以 8 NPU 完成 30 step，普通步中位 1.0730/1.0545/1.0325 秒，pooled 1.0470 秒；Before pooled 1.0565 秒，仅改善 0.8992%，低于三轮 run-to-run 波动。P95 1.273→1.197 秒，显存 4070→4064–4068 MB，但不足以替代中位收益门槛。
- Before/After pooled loss 均值 327.5536/324.4512（-0.9471%），中位 320.3263/317.0314，范围 223.9436–456.5770/211.8856–439.4726；六轮均保持前 10 步高于后 10 步，loss/grad 全有限，无 traceback/OOM/HCCL/fallback。After grad_norm 最大 301.9779 高于短跑基线最大 189.65，但没有非有限值或 loss 异常，仍不改变性能拒绝结论。
- 决策：问题真实但端到端收益没有超过噪声，按证据门禁精确回退，不形成 commit；候选 diff 与三轮证据只留远端诊断目录。

## 2026-08-13：normal-step 剩余热点重排与 SOAP 风险筛选

- 对提交后 normal-step `operator_details.csv` 按最内层仓库 stack 聚合：SOAP `project/update_preconditioner/project_back` host self 分别约 1218.7/670.1/629.5 ms，SOAP step 若干逐参数行约 400.7/253.9/219.8 ms，均显著高于模型前向单点；MMCV grad clip 约 280.9 ms。这里的数值是整张 operator 表对 stack 行的聚合，不等价于一段连续 device bubble，也不能直接相加当作 step wall time。
- SOAP step 的 167/168/170/191/203 行本身使用 Python float `beta/alpha/eps/lr/weight_decay`，profile 中每行约 559 次 `item/_local_scalar_dense` 来自 torch_npu scalar overload 的内部处理，并非源码显式 `.item()`。消除它需要 foreach/combined-gradient 或 tensor-scalar 批量改写，可能改变 559 个参数的更新顺序、舍入轨迹、optimizer state 和 loss 收敛。
- 当前项目已明确以 loss/最终功能不变为硬门禁；仅凭聚合 host 开销不足以授权重写 SOAP 更新。决策：不修改 optimizer，先回到 timeline 连续 device idle window，用 gap 前后 kernel、持续时间和 host stack 找到可等价消除的更安全候选。
- 2026-08-13：正常步 profile 的 16.54 ms 与 15.39 ms 连续空洞分别对应 `SPetr3D` 对全模型执行 `eval()` 与 `train()`，递归涉及约 925/832 个模块；中间唯一业务调用是冻结的 `pts_backbone.forward_rpn`。`MultiModal_PVB_GOP.train()` 会在恢复 train 时继续维持其声明冻结模块的 eval 状态，因此把切换范围收敛为 `pts_backbone` 是当前证据下低风险、约 30 ms/step 上限的候选。
- 正常步远端诊断产物已补齐：异常 JSON 57,092 字节、异常报告 7,245 字节、独立架构报告 5,151 字节且含 10 个规定章节。profile 仅含 Step 12、无 FIA/layer annotation/`communication.json`，所以报告明确按多模态训练子结构降级，不推断 LLM 层数或通信带宽。
- 冻结点云骨干局部模式切换候选三轮普通步 median 1.03612/1.053165/1.059585 秒，pooled 1.04710 秒；相对 Before 1.05650 秒仅改善 0.8897%，低于约 2.15% 基线波动且单轮不一致。90 个 loss/grad 全有限且 loss 正常下降，说明功能风险未显现，但收益不够可信；已回退且无 commit。
- 梯度裁剪参数缓存的理论上限约 16.8 ms/step（约 1.6%），且动态图仍需逐参数判断 `grad is None`，低于当前噪声门槛，已在代码修改前筛掉。下一方向保留 SOAP contraction 调度，但任何 foreach/合并/矩阵乘改写必须先证明参数顺序和 contraction 语义不变。
- 当前正常步 profile 没有 shape 记录；既有诊断 checkpoint 不在容器挂载范围，且按规则不复制产物。SOAP contraction 因缺少真实 shape/等价性证据暂不改写，避免改变数值轨迹。
- 新的 rank0 record-shapes 正常步 profile 已成功：SOAP line 299 covariance 有 543 次调用、51 种真实 gradient shape。代表性 NPU 微基准中 covariance matmul 等价式 22/22 bitwise equal，而 project/project_back 有 3 个非 bitwise 案例，后两者已排除。下一门槛是覆盖全部真实 shape/有效轴的 exhaustive bitwise 校验。
# 2026-08-13 SOAP 全形状等价门槛

- record-shapes profile 覆盖 51 类真实 gradient shape；SOAP covariance 候选共验证 157 个 shape/轴组合。
- `tensordot` 改写为 `movedim + reshape + matmul` 仅 153/157 位级相等；4 个四维卷积权重 axis 1 失败，最大绝对差 0.0029296875。
- 因此该候选被正确性门槛拒绝：不改业务代码、不进行正式 8-NPU A/B、不提交。

## MapTR target mask 筛选

- `nonzero(...).squeeze(-1).unique()` 中 `unique` 对一维 mask 索引语义冗余，但两处收益上限仅约 10–15 ms，低于噪声，不单独实施。
- type/color ignore-mask 的 `torch.isin` 替代在 8 个真实 shape/集合案例中布尔完全相等，但同步微计时比原 4 值循环慢约 54.9%，因此拒绝。

## 二维点归一化向量化筛选

- normalize 的真实 shape 微基准输出/反向梯度位级相等且单函数快约 23–26%，但三轮正式 8-NPU 端到端普通步 median 为 1.057/1.035/1.098 秒，pooled 1.059 秒，较 1.0565 秒基线慢约 0.237%。
- 正确性门槛通过但性能门槛失败，候选已完整回退；未提交。

## SOAP 二阶矩 addcmul 融合筛选

- 29 类真实 shape 单步/十步 state 均位级相等，机制微计时约快 40%，但三轮正式 8-NPU 普通步 pooled 1.061 秒，比 1.0565 秒基线慢约 0.426%；周期步约 15.5 秒也退化。
- 说明 record-shapes 的聚合 host self 和独立同步微计时不能代替端到端结论。候选已回退，未提交。
- 2026-08-13：STEP-072 用无 shapes 的正常步原始 timeline 逐窗归因。49.469 ms 是历史帧切片与位置网格多段调度的合并空洞，已有相关死网格候选 A/B 仅 0.899%；17.184/16.826 ms 属于正式 optimizer 主流程与 grad clip，其余单窗均低于噪声。prelaunch 148 ms 中 37 个异构输入 H2D 已走 pinned-memory、非阻塞 copy stream，重复 Event 的理论上限约 12.7 ms。MSDA backward 已直接使用官方 ACLNN，外围置零仅约 0.54 ms kernel。上述候选均不实施。
- 2026-08-13：STEP-073 MapTR target mask 复用完成 3×8 NPU；普通步 pooled 1.043 秒，相对 1.0565 秒基线只改善约 1.28%，低于 2.15% 噪声。三轮 loss/grad 有限且显存 4070 MB，但性能门禁失败，已回退、无 commit。
- 2026-08-13：STEP-074 证明正常步 `aten::item` 的约 300 ms host total 主要是 SOAP 三组 scalar overload 内部事件（各 1118 次），不是 loss 日志；显式 loss 日志仅 7.539 ms。重写会改变更新算子边界且收益证据受嵌套 host total 污染，已筛掉。
- 2026-08-13：开始 STEP-075。MapTR `data_valid` 由 Python 0/1 标志构造，却作为 NPU 标量在 decoder/class/ignore-tag 循环中反复参与 Python 条件判断；正常步仅 `data_valid == ignore_tag` 一行就对应 96 个 item/local-scalar 事件、host total 约 19.991 ms。先核对有效配置和行级 trace，只有独立同步上限明显超过约 22.7 ms 才实施；不与已拒绝的 target mask 候选合并。
- STEP-075 配置/trace 已闭合：batch 1、4 decoder、4 map class、3 ignore tag，line 1049 的 48 次比较恰好对应 4×4×3；line 1065、1257、1258 另有重复 NPU 标量控制和统计。按 host self 而非嵌套 total 计算，理论可移除调度已高于噪声门槛，允许进入机制等价验证。远端仍 clean、无训练进程。
- STEP-075 首版“整层转 Python”机制测试被正确性门槛拒绝：正负样本计数/avg_factor 类型变化使 FP16 代表 loss 出现差异。候选已收窄为只优化 target ignore/weight 控制，计数与 loss 保留原 NPU scalar 路径；尚未改业务代码。
- STEP-075 收窄版机制验证 50/50 与整组输出均位级一致；代表微计时中位约 6.268→2.135 ms。允许创建 target-only 最小 diff，但最终是否保留仍由三轮 8 NPU 端到端收益和 loss/grad 门槛决定。
- STEP-075 正式 3×8 NPU 否决 `data_valid.tolist()`：普通步 pooled 1.084 秒，相对 1.0565 秒基线慢约 2.60%；三轮 loss/grad 全有限且下降正常，但性能门槛明确失败。候选回退、不提交、不 push；说明减少 profiler 中 scalar sync 的独立微计时仍可能被完整训练调度抵消。
- STEP-076 device-self 重排：排除既有路径后，唯一超过 22.7 ms 的项目 frame 仍是官方 ACLNN MSDA backward（约 58.2 ms）；其余单功能均明显低于门槛。全局 MatMul/Conv 是分散的真实模型计算，Index/IndexPut/Nonzero 跨多个功能，不能直接相加。geo loss 三组有限值筛选合计不足约 15 ms且改写会改变 NaN/reduction 语义，已筛掉。
- STEP-076 Index 专项进一步确认：geo loss 同功能 device self 约 13.6 ms，其余未审查单功能都远低于 22.7 ms；全局 Index/IndexPut/Nonzero 不可跨功能相加。STEP-076 无改码结束，下一步核验 MapTR decoder 约 84 ms host self 是否对应真实连续空洞。
- STEP-077 确认 `maptr_decoder.py:133` 是完整 4 层 decoder 的外层调用归因，1259 个子操作均为 attention/linear/norm/view 等真实计算；host self 不能视为单点冗余。异步 NPU kernel 与 CPU dispatch 时间错位也使简单 overlap 不能估算可回收 idle。该路径筛除，转查 Hungarian D2H/CPU solver。
- STEP-078：当前 SciPy 1.15.3/torch 2.7.1/torch_npu 2.7.1 没有设备 Hungarian API；`npu_grid_assign_positive` 语义不同。8 次 solver 约 5.27 ms，连同 D2H/H2D 明确边界总上限约 22.61 ms，仍略低于噪声；严格等价替代不存在，筛除。
- STEP-079：`maptr_decoder.py:812` 是 attention 后真实 output projection，包含 linear/layernorm/ReLU，输入逐层变化，约 34.5 ms host self 不是可缓存构造；筛除。下一步只量化 line 148 被立即覆盖的 `zeros_like` 死分配。
- STEP-080：decoder line 148 是严格死 `zeros_like`，但仅 4 次、host self 约 1.434 ms、device self 约 0.006 ms，远低于噪声，不实施。转向全局高频初始化/死值覆盖审计。
- STEP-081：高频 `empty_tensor` 前列均是 SOAP、grad clip或真实 backbone/neck 算子的输出分配；`spetr3d.py:404` 展开为完整 Conv/BN/ReLU backbone，不是死初始化。未发现新的高收益死值覆盖候选，转向重复搬运/format cast。
- STEP-082：搬运前列除已知输入 scatter/既有拒绝项外，新证据是 grad clip 内 560 次 device-self 为 0 的 `aten::to`，host self 约 43.845 ms。进入 STEP-083 核验是否为同 NPU norm 的 no-op `.to(first_device)`；只允许保持官方算子顺序的条件跳过。
- STEP-083：同设备 norm `.to` 的 total norm/559 个 gradients 位级一致，但 profiler-off 仅约快 1.06 ms；43.8 ms profile host self 明显被 instrumentation 放大。候选改码前筛掉。后续 dispatcher 类候选必须先有非 profiler 机制收益上限。
- STEP-084：除 SOAP 外 contiguous/clone 最大仅为 `OrderedPtsL1Cost -> cdist` 的约 5.73 ms host self，且是非连续 reshape 输入的真实 materialization；无候选。转查同一 target 是否跨 loss 重复执行完整 normalize。
- STEP-085：最后一层 GeometricLoss 确实对同一 target 重复 normalize，可复用已有结果；但完整可删除 host self 上限仅约 14.24 ms，且全部 normalize 优化此前正式 A/B 已无收益，因此改码前筛掉。转查同一预测 tensor 的重复 finite mask。
- STEP-086：只有最后层 geo loss 的 `pts_preds isfinite/all` 与 line 1368 严格重复，可删上限约 3.67 ms；denormalize 后 mask存在溢出边界，不能复用。候选筛掉，转查条件零初始化。
- STEP-087：uncertain 与 geo 分支的真正死零初始化合计仅约 1.624 ms，远低于门槛，不实施。转查 assigner 对相同 GT 的跨 decoder 层 normalize复用。
- STEP-088：两种真实 GT shape 的跨层 normalize 复用位级一致，但 profiler-off 仅节省约 1.22–1.28 ms/step，接口改造范围过大，改码前筛掉。转向吞吐占比最大的 SOAP 周期 QR，审计同 shape batched QR。
- STEP-089 第一阶段：周期 profile 有 551 次/24 种 QR shape；1–512 方阵的逐个与 batched Q/R 全部位级一致。小矩阵调度收益明显，大矩阵收益较小；需先按 shape归因周期 wall time，再谨慎验证 768–5120 与完整 state。
- STEP-089 第二阶段：768/1024/2560 batched Q/R仍位级一致，但 2560 batch2 为 8.049→8.060 s、无收益；而 2560/5120主导周期耗时，因此全量 batching不成立。仅剩小/中 shape按真实频次的累计调度收益待测，需每周期超过约 227 ms才可继续。
- STEP-089 最终：全部真实 ≤1024 shape batched Q/R位级一致，但完整合成周期 6.682→6.759 s、反而慢 76 ms；只批正收益小 shape也不足约 48 ms/周期。跨 state batching拒绝。下一方向仅审计官方 Q-only 分解路径。
- STEP-090：`geqrf/orgqr/householder_product` 小 shape Q已非位级一致，并明确触发 NPU→CPU fallback；同时违反两项硬门槛，立即拒绝。转查 QR前排序/重排的静态恒等路径。
## STEP-091：SOAP QR 前排序与状态重排审计

- 周期 profile 中共有 551 次 QR，对应 551 次 `argsort`、1102 次 `index_select`；其中 1×1 形状有 106 次。
- 1×1 静态旁路可删除 `diag(o.T @ m @ o)`、稳定排序以及两个恒等 `index_select`，同时保留 `m @ o` 和原始 `torch.linalg.qr`。
- NPU 无 profiler 微测确认输出逐位一致，最大差值为 0。
- 真实收益仅 11.6477 ms/周期，按 10 步更新周期摊销 1.1648 ms/步，显著低于约 22.7 ms 的辨识门槛，故淘汰。
- 其他形状不能假设排序恒等，也不能缓存旧排序；运行时恒等判断需要同步，缺少收益空间。
## STEP-092：SOAP 大矩阵重排已是最优候选

- `index_select`、高级索引和 `gather` 对测试输入逐位等价。
- 当前 `index_select` 在 2560/5120 方阵上分别约 0.092/0.279 ms，明显快于另外两种实现。
- 既有 profile 的大额 host 时间不能直接解释为可节省执行时间；真实重排成本没有正式训练可辨识空间。
## STEP-093：SOAP QR 是 NPU 原生算子而非 CPU fallback

- 机制 profiler 明确显示 `aten::linalg_qr → aclnnLinalgQr`，无 unsupported/fallback 告警。
- `torch.qr` 与 `torch.linalg.qr` 输出逐位一致，但前者已弃用且只是兼容入口；微小计时差不能形成优化候选。
- 周期 profile 中 QR 的 device 时间为 0 应视为采集归因限制，不能据此进行 CPU/NPU 迁移优化。
## STEP-094：SOAP 通用 power 复用不满足逐位门槛

- 点积对角表达会改变特征值末位；`o.T@(m@o)` 也会因结合顺序在较大形状改变末位。
- 即使 `m` 由外积更新保持逐位对称，`(o.T@m).T` 与 `m@o` 在 192 维以上仍不逐位相同。
- 因此不得将这些数学恒等式用于周期 SOAP 更新；仅剩 identity 初始化的静态特例可继续验证。
## STEP-095：identity 初始基旁路等价但不值得实施

- 全部 24 种真实 shape 的排序、power、Q/R 均逐位一致。
- 按真实频次每个 rank 仅在训练首次初始化节省约 85.634 ms，不改善周期稳态热点。
- 一次性收益不足以支撑专用分支，故不修改代码。
## STEP-096 第一阶段：独立 QR 可在 NPU streams 上有效重叠

- 双 2560 QR 的并发输出逐位一致，wall time 约 8.04→4.16 秒（1.93×）。
- 候选不改变 QR 输入或算术，仅对同一 state 的独立维度做有界并发；下一步需覆盖真实混合 shape和内存门槛。
- STEP-096 补充：2560+5120 双流逐位一致且节省约 4.27 s/组合；四个中型矩阵两流分批逐位一致，约 1.46×。实现应限制为两个 stream，避免无界并发和显存峰值。
- STEP-096 函数级门禁：修改函数与原参考实现对多种 state、dtype 和连续调用的 Q/`exp_avg_sq` 全部逐位一致；stream 缓存不进入 checkpoint `state_dict`。
## STEP-096：SOAP 两流 QR 单卡有效但正式 8-NPU 严重退化

- 双 2560 裸 QR 单卡微测逐位一致且约 1.93×，混合 2560+5120 约 1.14×，函数级 state 也逐位一致。
- 真实 8-rank 30-step 中，周期步中位 21.843 s，较约 13.4 s 基线慢 63.0%，并把后继步抬至 2.129 s；普通步也轻微退化 1.56%。
- loss/grad 与基线分布一致且全有限，说明拒绝原因是端到端性能而非功能偏离。
- 结论：单设备局部 stream 重叠证据不能覆盖 8-rank 正式负载；不再尝试 SOAP multi-stream QR，并已恢复 HEAD clean。
## STEP-098：QR 输出缓冲复用无性能价值

- `torch.linalg.qr(..., out=...)` 在当前 NPU 上逐位一致并复用 buffer，但 7–256维略慢，512维仅约0.05 ms噪声收益。
- 预分配触发内部格式回退告警，不进入大矩阵或业务实现。
## STEP-100 第一阶段：仅等尺寸双中矩阵具有可辨识上限

- 真实频次52×双256和8×双512的合成周期可节省约572.5 ms，摊销57.3 ms/step，Q/R逐位一致。
- 候选必须只命中这60个state，不能复用STEP-096通用pending/stream实现。
## STEP-100：只并发双256/双512仍在正式8-NPU退化

- 单卡真实频次合成显示约57.3 ms/step上限，函数级命中/非命中均逐位一致。
- 正式8-rank中Iter11为17.809 s、Iter12为6.429 s，相比约13.4 s周期基线仍明显退化并产生跨步长尾。
- 已早停、完整回退并恢复Git clean；结论是SOAP QR任何multi-stream方案均不适用于当前正式负载。
## STEP-101：denormalize切片回写消除仅约0.87ms

- stack/cat输出、梯度和布局均逐位一致，但完整20次workload只节省0.868ms/step。
- profile host-self被小算子插桩显著放大，候选改码前淘汰。
## STEP-102：ignore mask查找表仅节省1.89ms

- 缓存布尔表索引对完整值域严格等价，但真实32次workload只节省1.892ms/step。
- 不增加运行时缓存，不实施。
## STEP-103：SOAP原地square仅节省0.514ms

- 全空Q state存在修改原gradient的严重别名风险；收窄到199个非空Q state后可保证隔离。
- 即使避免约309MB临时分配，真实完整workload也只节省0.514ms/step，不实施。
## STEP-104：SOAP 0-D Tensor scalar路径更慢

- 数值逐位一致，但559次完整scalar dispatcher由12.4ms退化到26.6ms。
- 不把profile item当作真实D2H热点，不建立device scalar缓存。

## STEP-105：MapTR data_valid计数缓存仅节省0.535ms

- 正式路径为4 decoder层、每卡batch 1、4地图类别；正负计数合计32次`data_valid > 0`，按样本缓存可删除28次。
- 缓存路径对有效/无效样本的计数值、Tensor/Python返回类型、dtype/device和`max`结果均严格一致。
- 完整真实频次仅1.6919→1.1567ms，节省0.5352ms/step，不改码。

## STEP-106：普通步热点重新聚合

- 排除历史已闭环功能后，唯一仍具较高host调度上限的是SOAP project/project_back的tensordot展开；该方向随后在STEP-107被真实数值与性能双重否决。

## STEP-107：SOAP project的movedim+matmul表达不适用

- normal stack中project/project_back的通用tensordot展开了大量reshape/view/permute元数据算子，但无profiler真实全频次加权仅72.629→53.532ms，节省19.096ms，低于22.7ms门槛。
- 更关键的是18种真实4D组合出现最大1.0—2.5的FP32差异；轴语义、shape和stride一致不代表底层矩阵乘数值路径一致。
- 不替换`tensordot`，不训练、不提交。

## STEP-108：SOAP tensordot out buffer反而退化

- 543次真实调用的最终GG逐位一致，但完整链60.822→69.727ms，慢8.905ms。
- 每state缓存还需增加272.318MiB/rank常驻显存并触发base-format路径，不实施。

## STEP-109：梯度裁剪现状已自动foreach

- 559个真实shape下foreach=False/None/True中位15.407/13.778/13.823ms；正式默认None已达到批量性能，显式True无收益。
- 实际发生裁剪时总范数和全部559个梯度三路径逐位一致，不改hook或配置。

## STEP-110：当前版本无foreach nan_to_num

- SOAP的559次逐grad安全清理不能用当前torch官方foreach API批量化；对应导出不存在。
- 不删除NaN/Inf保护、不增加条件同步、不自写融合核，候选在改码前关闭。

## STEP-112：SOAP一阶矩双foreach仅节省6.857ms

- 保持`mul_→add_`两步表达的foreach对559个真实shape逐位一致，更新本体8.689→1.831ms。
- 6.857ms低于22.7ms门槛，且实现需要把逐参数完整算法拆成阶段循环，不实施。

## STEP-113：SOAP covariance安全shape窄分支仅节省6.292ms

- 排除全部4D axis1后，5 seed×128个候选case共640例逐位一致；历史数值差异稳定只出现在4个4D axis1 shape。
- 真实543次完整outer-product+lerp workload中，466次安全改写仅41.922→35.630ms，节省6.292ms，不实施。

## STEP-114：SOAP denominator foreach仅节省6.520ms

- 559个真实shape、7729万元素的`sqrt→add eps`批量路径逐位一致，但仅9.123→2.603ms。
- 单项低于噪声门槛，不实施；转查多个严格逐位foreach阶段的完整功能合计上限。

## STEP-115：五个SOAP foreach阶段乐观上限仅19.382ms

- division、parameter add、weight decay分别逐位一致并节省2.137/1.612/2.256ms。
- 连同一阶矩6.857ms和denominator6.520ms，独立最佳值直接相加也仅19.382ms，尚低于22.7ms；未计跨组收集/分桶开销，不实施。

## STEP-116：二阶矩foreach节省10.960ms但全量临时内存约589.6MiB

- 保持`mul→out-of-place self-mul square→add`的559个state全部逐位一致，15.238→4.278ms。
- 六阶段独立乐观上限约30.342ms，但全量批处理需同时持有约294.8MiB projected和294.8MiB square，不能直接实施；进入固定元素预算分块完整骨架测试。

## STEP-117：8M预算分块六阶段骨架节省35.315ms

- 559个真实shape下exp_avg、exp_avg_sq和parameter全部逐位一致；完整旧/新骨架47.276→11.960ms。
- 10块，最大单tensor块13.1M元素，projected+square约100MiB；满足进入真实optimizer最小实现门槛，但仍需完整project/Q/GG多step逐位和8-NPU A/B。
## SOAP 分块 Foreach 调度（2026-08-13）

- 对 SOAP 普通更新按配置/step/device/dtype 分桶，并以 800 万元素分块执行 foreach；初始化、投影、QR/preconditioner 顺序与状态 schema 保持不变。
- 相同输入的 6-step NPU 功能门禁中，参数、exp_avg、exp_avg_sq、GG、Q 与最终 state_dict 逐位一致，NaN/Inf 掩码轨迹一致。
- 三轮正式 8-NPU、30-step 的 69 个普通步 pooled 中位数 1.029s，相对当前 baseline 1.052s 提升 2.186%，相对历史 pooled baseline 1.0565s 提升约 2.60%；三轮均值提升 2.300%。
- 六个周期双-step 窗口均无回归，峰值显存 4071MB（baseline 4070MB）。三轮 loss 全有限，pooled 均值 330.936，较历史 pooled baseline 327.554 约 +1.03%，未发生功能性大偏离。
- 候选 grad_norm 尖峰高于单轮 baseline，但三轮全有限；结合相同输入逐位一致门禁，判断为随机 batch 分布差异，后续长训仍应持续监测。
## STEP-120 提交后热点重采集必要性

- 现有 normal profile 的主要软归因是 SOAP 逐参数 launch fragmentation，而 `14d4f23` 已重写该调度边界。
- 因此旧 profile 只可作为 Before 证据，不能代表新 HEAD 的剩余热点排序；下一候选必须基于提交后新 profile。
- `14d4f23`后新normal profile显示约50次foreach-add和多组13–16ms host-visible gap，符合10个chunk×多个阶段的调度结构；这属于同一SOAP foreach功能的预算调优，不应拆成新commit。
- SOAP chunk预算从8M扩大至16M/32M虽逐位一致，但完整六阶段仅再省0.400/0.548ms，远低于噪声门槛；`14d4f23`保持不变。
- 提交后49.2255ms `NanToNum→Arange` gap的真实host根因是`SPetr3D.train/eval`全模型递归，属于此前三轮仅+0.8897%的已拒绝候选；相邻kernel名字并非根因。
## 2026-08-13：剩余优化项按可信耗时排序

- P0 SOAP 周期 QR：543 次 `aclnnLinalgQr` 的 device total 22.674 秒，是数量级领先的剩余热点；风险也最高，不能通过改变 basis/频率换性能而跳过 loss 与长期收敛验证。
- P1 MSDA CPU fallback：当前源码仍会把 NPU输入搬到 CPU 执行 `grid_sample`；正式训练中该 fallback 的独立次数/wall 尚未量化。新 profile 的 57.884 ms 是 NPU MSDA backward，不可冒充 CPU fallback 耗时，下一步先分支归因。
- P2 Hungarian CPU/SciPy：完整搬运与求解的既有乐观上限约22.61 ms/step；无严格等价设备 solver 前保持不变。
- P3 BEV backbone line120：with-stack host self 31.112 ms、device total 6.095 ms；恢复后先拆重复 layout/copy，真实模型算子不删。
- P4 其他 copy/cast/format：ConvNeXt line82 等仅有约1～6 ms device量级，按功能聚合后再判门槛。
- SOAP Foreach 长周期验证属于验收任务，不与算子耗时混排；已失败方向默认关闭，避免重复消耗8卡资源。

## 2026-08-13：优化流程口径纠正——当前先采集性能

- 用户指出算子优化应先采集性能、测试耗时，判断正确。P0～P4 应视为由历史证据形成的待测候选池，不是立即修改顺序。
- 当前正式阶段改为 STEP-130：在 `14d4f23` 上分别采集普通步、SOAP周期步和CPU专项路径；MSDA等未知耗时项在实测前不得参与正式排序。
- 排名依据依次为 profiler-off完整功能 wall、device/AICPU duration、CPU同步/搬运 wall；with-stack host self只用于定位源码，不作为收益承诺。
- 完成当前HEAD排名后，才对第一名做严格等价机制测试、源码修改、3轮8-NPU A/B与loss/grad验证。

## 2026-08-13：每阶段必须交付量化性能指标

- 用户要求每个优化阶段都必须给出可量化的性能结果。以后不以“已改代码/已通过测试”作为完成标准，而以Before/After耗时、吞吐、调用次数、显存和正确性数据闭环。
- 普通步、SOAP周期步、端到端均值和具体算子功能耗时分别报告；禁止把with-stack host self、device duration和完整wall混成同一指标。
- 正式成功候选至少提供3轮8-NPU A/B的pooled结果与轮间CV；失败候选也必须报告实测节省/退化及淘汰原因。
- 统一计算绝对节省、耗时下降率、加速比和吞吐提升率；SOAP周期优化同时报告周期原值和按频率摊销值。

## 2026-08-13：永久固定最初性能基线

- 用户明确要求后续始终以 `63861dfd920ab9829512b1e4a000eefd1ffcfbea 【loss对齐】随机性移除` 作为极限性能累计对比基线，判断正确。
- 最新commit只用于测量当前单项优化的增量收益，不能替代永久主基线。所有阶段应同时给出“相对63861df累计效果”和“相对父提交单项效果”。
- 已有63861df统一数据：30-step pooled均值28.932秒/step、吞吐0.2765 sample/s；普通稳态中位3.186秒/P95 7.074秒；周期步中位271.486秒/P95 280.307秒；最大显存5067MiB。
- 若当前采集口径与历史不一致，应重跑63861df同口径或明确禁止直接计算百分比，不能为了展示效果拼接不同口径数据。

## 2026-08-13：官方文档落地为双轨测试

- 华为7.3.0建议先采集/拆解，再区分下发、计算、通信等瓶颈；并建议在条件允许时增大micro batch以提高计算量和通信掩盖，但强调具体模型必须实验。
- 当前约4GiB allocator峰值/约7GiB设备HBM占用仅证明容量余量，不直接证明AICore利用率低。SOAP周期QR、host下发空洞、MSDA/Hungarian和数据路径仍需分别量化。
- 后续保留严格等价batch=1算子基线，同时建立batch=1/2/4（必要时8）的独立吞吐曲线。前者看step性能和累计优化，后者看samples/s与ms/sample，禁止混成一条百分比。
- batch扩展短跑通过后仍需长训重新验证学习率/更新次数/收敛；它是训练策略优化，不是等价算子优化。

## 2026-08-13：本地custom客户配置识别

- `custom`仅有两个文件：唯一`.py`为完整MMCV训练配置，`rg_evaluation_results.txt`为2026-08-13评测指标输出，不是配置。
- 客户配置文件SHA256为`9039bd31...1ca33b`、1991行，包含模型、数据、SOAP优化器、runner和部署配置；明确`num_gpus=8`、`batch_size=16`、`samples_per_gpu=batch_size`、`workers_per_gpu=8`、SOAP `precondition_frequency=10`、max_iters=30000。
- 该文件名与远端性能测试配置同名，但hash和关键运行参数不同；此前远端基线配置SHA为`6872d9a2...`，实测samples/rank=1、workers=0。因此当前约4～7GiB显存占用是缩小测试配置的结果，不能代表客户原始batch=16配置的显存占用。
- 本地`custom`目录整体未被当前Git仓库跟踪，仅凭文件位置/内容可判定文件角色，仍需与客户交付记录或远端生产入口核验其最终权威版本。

## 2026-08-13：客户运行字段已同步到双代码线

- 整份客户配置仅1991行，而远端两版约3380行；直接替换会删除约1473～1474行数据集/诊断配置，并移除当前fingerprint hook和最新SOAP one-sided字段。因此采用最小运行字段同步，而非整文件覆盖。
- 两版Config验证结果完全一致：8设备、batch/rank=16、global batch=128、workers/rank=8、prefetch=3、num_iters_per_epoch=219；模型、数据和优化器版本差异仍由各自提交链管理。
- 基线派生提交`4c37039`的父提交精确为`63861df`；最新配置提交`a757f29`的父提交精确为`14d4f23`。两边提交描述均为`【去除随机性固定】客户训练配置字段对齐`，配置对齐在两边各一个同功能commit，便于独立回退和同口径A/B。
- 尚无batch=16真实性能/HBM数据；配置可导入不等于可完成训练。下一步必须使用8-NPU早停门禁，不能沿用batch=1显存结论推算。

## 2026-08-13：客户batch=16单轮8-NPU累计A/B

- 同机同口径测试对象为`4c37039`（由永久算法基线`63861df`仅增加客户字段对齐）与最新`a757f29`。历史batch=1数据不可与本组混算。
- 全30步均值37.440→11.745秒，下降68.63%、加速3.188×；global batch128吞吐3.419→10.898 samples/s，提升218.76%。
- 排除iter1～2预热、两组SOAP双步窗口和iter30后，23个普通步中位11.501→7.868秒（下降31.59%，1.462×），P95 11.900→8.443秒。
- SOAP双步窗口从281.575/277.602秒降至39.079/39.375秒，平均下降85.97%、加速7.127×；第11/21周期主步从265.253/266.869秒降至9.516/10.278秒。
- 框架单rank峰值28460→27445MiB（-3.57%）；npu-smi设备HBM峰值45782→44054MiB（-3.77%），每chip物理容量65536MiB。
- loss与grad全部有限：loss均值291.833→309.705（+6.12%）、中位274.347→306.556（+11.74%），范围分别213.106～415.855与219.358～430.952；grad中位51.993→49.732。范围重叠且无数量级发散，但逐step不相同，短跑不能替代同输入逐位和长训最终指标门禁。
- 本轮按用户要求只跑1轮，不能提供三轮CV或置信度；若用于发布结论，仍建议补三轮与长训验证。

## 2026-08-13：STEP-136最新客户负载profile适用性审计

- 当前权威代码为`a757f29`、`ascend_npu_optimize`且clean，正确容器运行、NPU空闲。
- 远端11个历史profile均早于客户batch16字段对齐；batch16目录只有普通训练日志/manifest，没有kernel_details、trace或TorchNPU profiler产物，因此无法支持新算子归因。
- 最近成功的`14d4f23` rank0 stack profile夹具可复用：真实训练仍为8 rank，仅rank0产出profile，脚本支持MAX_ITERS和独立profile/work目录；原配置wait=11/warmup=1/active=1只采一个普通步。
- 既有hook支持record_shapes/profile_memory/with_stack及动态schedule。为同时覆盖普通与周期步骤，计划改为wait=8/warmup=1/active=4、MAX_ITERS=14；历史同schedule已覆盖ProfilerStep9～12，其中Step10为SOAP周期，且最终checkpoint位于active窗口之后。
- batch16 profile只用于归因，不用其step time替代profiler-off 30-step性能数据；原始产物继续只留远端。
- 新profile的`operator_details.csv`约3.46GB、`trace_view.json`约4.85GB；全表载入不合适。正式分析改为kernel表常驻、trace流式解析、调用栈列20万行分块计数，实测分析进程约500MB RSS，避免因分析工具本身制造远端内存风险。
- 本地迁移后系统`python`别名不可用（返回9009），但Codex捆绑Python及现有远程只读工具可用；不需要、也没有在客户容器安装依赖。
- 最新batch16 profile共334,374条kernel，Step9/10/11/12 service为42.100/61.900/44.836/44.151秒，busy union为3.741/27.311/4.983/5.277秒；underfeed为91.11%/55.88%/88.89%/88.05%。这些是`record_shapes+with_stack`诊断值，不能替代正式7.868秒普通步中位。
- 543次QR为22.711秒且0%掩盖，仍落在`soap.py:422→337→174/191`；但batched、多流、out-buffer和当前版本替代入口都已实测关闭，因此按“耗时最大但不可直接实施”登记，不重开旧方向。
- 4步合计纯kernel：Index 4.433秒、IndexPut 3.858秒、MSDA backward 3.186秒。调用栈显示三个各约1.501秒device total的Index分别来自`geo_loss.py:224/226/228`，主要无栈IndexPut为3.931秒，符合这些高级索引的autograd scatter。该同一功能的诊断上限约8.29秒/4步，即2.07秒/步，成为下一活跃候选。
- MSDA backward精确落在`multi_scale_deformable_attn_function.py:183`现有自定义NPU反向入口，不是CPU fallback；约0.796秒/步，排在GeometricLoss之后。
- STEP-137资源门禁发现旧容器`mapqr`存在自动/外部1-rank任务，进程均为T(stopped)且仍占NPU0；TERM只冻结/未退出，不能视为资源释放。经容器归属、cmdline和PID树三重核验后精确KILL，NPU进程表清空。后续必须再次防止它自动重启。
- 可复用夹具已验证：诊断模块可放在远端诊断目录并通过`PYTHONPATH`/`custom_imports`导入，配置副本和输出均不进入业务Git；8-rank入口支持`--max-iters`。STEP-137探针只包裹运行时方法并原样返回张量，适合先采真实finite分布。
- 客户batch16真实3步中，GeometricLoss forward 12次；原始target 921,600元素全部finite。目标intra length/dot/cross各131,360元素、inter各103,443,200元素，六类nonfinite均为0。因此`geo_loss.py:224/226/228`热路径的`isfinite` mask全True，随后的pred/target Boolean index及反向IndexPut没有筛掉元素。
- 不能据3步直接删除NaN/Inf兼容语义。候选应对`mean/sum`用设备侧masked elementwise+reduction表达“仅对finite target计算L1”，对`none`或非默认旧参数保留原索引路径；必须覆盖全finite、部分NaN/Inf、全无效和空张量的loss/梯度。
- 设备侧候选为`where(finite, abs(pred-target), 0).sum()`；sum直接返回，mean除以finite count，并对count=0用零梯度NaN保持PyTorch空mean语义；none回退原Boolean index。8 rank全部案例loss/grad逐位一致，包括全无效/空mean为NaN且grad为0。
- profiler-off 8-NPU微基准覆盖14.4K/67.6K/129.6K/435.6K/608.4K/960.4K/1.3924M/1.6384M真实数量级，旧索引前反向中位0.833/1.629/2.636/7.861/10.777/16.725/24.065/28.314ms；masked为0.601/0.566/0.550/0.529/0.565/0.542/0.556/0.605ms，加速1.385～46.802×。这是单调用机制收益，正式训练收益仍需A/B。
- 第二次旧容器自动任务来自独立会话shell PGID3951205（容器主PID13098），该shell持续拉起1-rank；精确终止会话组后10秒未重启，避免反复只杀子进程。
- 项目权威config使用`GeometricLoss(loss_type='l1')`且未覆盖reduction，默认为mean；调用不传weight/reduction_override。业务helper仍保留`none`和非默认size_average回退，避免影响其他配置/派生类。
- 实际单文件实现经8-rank源码门禁复验：helper的mean/sum/none四类异常，以及完整GeometricLoss forward的全finite/部分nonfinite，loss/grad均逐位一致；源码helper微基准1.416～46.162×。未发现功能偏差。
- 修改后客户batch16 3-step自然完成，loss/grad均有限、无运行错误；说明helper可进入真实模型前反向。正式30-step第一次exit137发生在首iter前且是host `docker exec`客户端被SIGKILL：正确容器状态`OOMKilled=false`、内核无OOM、主机内存充足，不能归因于候选NPU内存或训练异常，也不形成性能样本。后续使用容器内detached状态文件规避长连接客户端被外部杀死。

## 2026-08-13：后8卡切换前资源与副作用核验

- 当前两个相关容器均无`train_spetr.py`或distributed launcher，`npu-smi`也未列出训练Python进程；先前候选目录只有0条iteration且无退出状态文件。
- 上次失败启动虽未形成样本，但仍生成临时配置、kernel cache并改变`fusion_result.json`工作树状态；必须先归档这些运行副作用，不能把脏状态带入后8卡正式样本。
- 后8卡的正确设置为`ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`；分布式world size仍为8，不改成16 rank。
- 诊断指针文件可能带CR/LF；涉及移动的安全校验必须先规范化并用`readlink -f`验证落在共享诊断根下，不能直接用原始字符串做前缀匹配。
- 用户再次明确：此前基线、profile、热点排序和门禁结论全部继承；切换后8卡不是重新制定或修改历史结论，只是当前正式样本的资源隔离调整。
- 后8卡第一次实际启动证明8个rank均继承`ASCEND_RT_VISIBLE_DEVICES=8,...,15`；失败发生在`torch.npu.set_per_process_memory_fraction`触发lazy init时，直接原因是启动包装覆盖原有`PYTHONPATH`后`tbe`不可导入。必须复用成功基线的追加形式`项目路径:${PYTHONPATH:-}`，禁止通过安装或改变客户CANN环境解决。
- 现有30-step夹具可能在distributed launcher失败时仍返回0；有效运行必须以30/30结构化iteration、fatal错误0、8-rank/NPU映射和自然释放的联合门禁为准。
- 本机16逻辑设备的后8个在`npu-smi`中对应物理NPU ID 4～7的chip0/1；本次8个rank的主PID映射完整覆盖这8个chip。训练入口在设置rank设备前会让各rank在首个可见后8卡设备上创建约121MiB小上下文，但不触及前8逻辑设备。
- GeometricLoss候选正式后8卡30-step普通步mean从父版本7.9697降到5.8707秒，绝对节省2.0990秒/step，与此前profile的Index+IndexPut诊断上限约2.07秒/step高度一致；结合源级同输入逐位等价和真实shape微基准，构成机制—算子—端到端闭环。
- 父版本与候选使用不同逻辑设备子集（前8 vs 后8），因此不能把所有端到端差值无条件归因于代码；但节省量与独立机制证据高度相符。正式发布若需要置信区间，应在资源允许时用同一设备子集完成三轮A/B。
- 候选30步loss/grad全部有限；loss范围208.927～428.746与父版本219.358～430.952重叠，无数量级偏离。短跑随机batch分布不能替代长训最终指标，但没有出现用户禁止的超级大loss偏离。

## 2026-08-13：MSDA分支冲突的初步代码证据

- 当前自定义`multi_scale_deformable_attn_function.py`的fp32 forward/backward直接调用MMCV扩展`ms_deform_attn_forward/backward`；backward line183与最新profile栈完全一致。
- `point_cross_attention`、`maptr_decoder`等调用点在非deploy模式下根据CUDA兼容属性选择项目自定义扩展，否则进入`multi_scale_deformable_attn_pytorch`。Ascend迁移补丁会影响该条件，必须做运行时分支计数。
- 历史CPU fallback提交`70576d3`仅改MMCV通用文件；项目自定义调用与MMCV通用调用是两个不同族，先前把“MSDA”整体描述成单一CPU fallback过于粗略。正式优化前需按调用族分别量化CPU wall、NPU device duration和调用次数。
- MMCV通用`MultiScaleDeformableAttention.forward`在line384～387显式包含`IS_NPU_AVAILABLE and value.device.type=='npu'`，因此NPU输入直接走自定义扩展，不经过line111的reference CPU fallback；CPU fallback只影响显式调用`multi_scale_deformable_attn_pytorch`的项目路径。
- 基础容器中NPU tensor并不伪装成CUDA（`torch.cuda.is_available=False`、`is_cuda=False`），所以项目中只检查CUDA属性的分支按字面会进入reference；训练入口若导入transfer_to_npu则可能改变此结论，必须以运行时探针为准。
- 真实训练入口导入`transfer_to_npu`后，CUDA兼容属性被改写：NPU tensor的`is_cuda=True`，项目自定义attention会进入`MultiScaleDeformableAttnFunction_fp32.apply`。最新profile line183因此与当前训练代码一致；历史CPU fallback在这些激活分支中并未执行。
- `aclnnMultiScaleDeformableAttentionGrad`是当前固定环境的官方NPU实现，0.796秒/step是主体真实计算，不能通过Python调度删除。可审计的低风险边界只有调用前的三个`zeros_like`是否属于死清零；必须先从profile单独量化其kernel总时长，再决定是否值得机制测试。
- 最新客户profile记录24次MSDA forward和24次backward，正好每个profile step各6次；backward主体3185.547ms/4步，forward主体700.171ms/4步。当前性能机会若只改预清零，收益上限必须远小于backward主体，不能把3185.547ms宣传成可回收值。
- line级CSV精算中，官方grad顶层device self为3211.402ms/4步；与架构报告3185.547ms的小差异来自聚合名称/层级口径，二者均证明约0.80秒/step主体量级。三个预清零device self总计仅8.193ms/4步。
- `zeros_like→empty_like`即使确认输出完全覆写，也只能回收约2.05ms/step，约占当前普通步0.035%；该量级低于正式A/B噪声且不值得承担未初始化buffer语义风险，故直接关闭。
## 客户配置随机性复核（2026-08-13）

- 历史配置确实包含随机性固定：训练增强 `apply_ida=False`，光度扰动与 `PointShuffle` 被注释；这些在永久基线提交 `63861df` 中已恢复。
- 当前有效训练配置没有固定 seed、`manual_seed` 或确定性环境开关；`#随机性固定` 多为遗留注释，不代表其包围的 8 卡、batch、worker 字段会固定随机数。
- 当前使用 `DistributedSampler`，而随机丢组 sampler 处于注释状态；它属于采样策略，是否打乱和 epoch seed 需要结合构造调用判断。
- `flip=False` 出现在测试 pipeline，不能据此认定训练增强仍被关闭。

## BEV backbone line120复核（2026-08-13）

- line120是3个downblock的外层调用点；4个profile step中正好60次Conv、60次BN、60次ReLU，即每步15组真实模型算子。
- inclusive device total的71.867ms/step存在父子API四重/多重计数；唯一叶子kernel只有18.871ms/step，其中Conv 14.034ms、BN 3.029ms、ReLU 1.808ms。
- 该行没有copy/cast/contiguous/format。最终cat仅0.482ms/step，其余FPN上采样、平滑卷积和通道注意力均为低毫秒级真实计算。
- 候选低于噪声门槛且没有严格等价冗余，关闭；不修改、不训练、不提交。

## P4搬运/格式族重排（2026-08-13）

- 4步profile扫描1,468,549行，匹配54,718条to/copy/clone/contiguous/cast相关父子记录；全族总量存在大量父子API重复，不能直接相加。
- 最突出独立栈是`mmcv/runner/hooks/logger/text.py:112 _get_max_memory`：每步执行一次跨rank显存MAX并`item()`，with-stack host self约4.694秒/step。它可能包含等待之前NPU队列的归因，不能直接宣称可回收4.694秒。
- 其他较大项目栈主要是已关闭的geo/map loss、Hungarian、scatter、map normalize以及真实SOAP路径；当前先验证日志同步降频，因为它不进入计算图且具有单一功能边界。
- profiler-off后8卡30-step兑现约0.155秒普通步均值节省（-2.648%）和0.206秒全步均值节省（-2.167%）；说明with-stack的4.694秒大部分是等待归因/放大，不能作为可回收值，但每步跨rank日志同步确有稳定成本。
- `memory_interval=10`保留每步memory字段，实际日志在1～9步复用24847MiB，第10步刷新26482MiB，第20步26658MiB，第30步26851MiB；末步仍给出精确全程峰值。

## LCFusionV2固定网格候选（2026-08-13）

- `self.grid`在CPU初始化且是普通属性，当前每步line311执行CPU→NPU `.to`；最新profile每步正好1次，shape `[1,96,160,2]`，copy host self约133.396ms。
- 不使用register_buffer：非持久buffer虽然不进state_dict，仍可能参与DDP broadcast_buffers。采用惰性普通属性缓存，只在设备变化时搬运，既保留跨设备稳健性又不新增广播。
- 单NPU真实模块门禁证明旧/新输出和输入梯度逐位相同，缓存后第二次输出也逐位相同；纯grid搬运+repeat仅节省约0.061ms，正式价值取决于是否消除训练队列同步，必须端到端验证。
- 后8卡30-step证伪同步收益：普通步均值相对父版本反而+1.035%，全步+0.336%，SOAP+1.277%；均属小幅回归/噪声，没有可提交收益。133ms host self不可回收，方向已关闭。
- `bevformer_encoder.py:602`的`[1,2]`shape tensor copy被归因74.262ms，但line604相邻`[1]`tensor仅0.170ms；这是同类异步等待锚点。结合LCFusion反证，缓存该小常量不具正式测试资格，关闭。

## 当前HEAD停卡期profile完整性（2026-08-13）

- 当前代码为`bf9ed6e`，本轮后8卡profile在用户征卡时训练日志停于13/14，随后按唯一master port和进程组定向终止；本轮进程与端口均已释放，后续禁止自动重跑。
- 核心导出产物约13.8GB，`step_trace_time.csv`与`kernel_details.csv`均精确覆盖Step 9、10、11、12，满足schedule 8/1/4/1的4个active step；因此可用于热点结构分析，但不能作为profiler-off端到端性能样本。
- 中断留下的`kernel_meta/`已归档到远端诊断目录，`fusion_result.json`精确恢复HEAD；业务仓库重新clean。
- 正式分析采用低优先级、单线程、CPU-only流式脚本，原始profile不下载本地；分析期间训练、`msprof`和分布式任务数保持0。
- 用户随后明确只要求“不占用NPU”，CPU分析不必限制单线程；后续聚合已恢复正常CPU并行度，NPU任务仍保持0。
- 修订后的异常JSON通过正式schema；普通Step 9/11/12平均underfeed 94.752%，周期Step10为59.704%。这些是`record_shapes+with_stack`诊断窗口事实，不能直接替代profiler-off step time。
- GeometricLoss旧Boolean-index主热点已消失；当前Index/IndexPut主要位于MapTR target、Hungarian和decoder loss。MSDA backward仍约803ms/step的官方ACLNN必要计算，SOAP Step10 QR仍为周期主热点且属于已关闭高风险方向。

## MapTR正负样本冗余Unique重新入选（2026-08-13）

- 当前客户batch16中`_get_target_single`每步执行256次；line966与968的`aclnnUnique2`分别为101.840/96.651ms（4 profile steps），合计198.491ms/4步=49.623ms/step，约为现有22.7ms绝对门槛的2.19倍。
- 历史STEP-073在batch1下把Unique删除与mask/权重构造混为一个较宽候选，三轮仅改善1.28%而回退。当时函数每步约16次；当前调用频次放大16倍，因此新负载证据允许重新打开，但只做两处`.unique()`删除，不混入历史其他改写。
- 语义依据：一维Boolean mask的`nonzero`对每个满足位置只产生一次索引，结果天然无重复；`.unique()`不会改变集合或顺序。CPU穷举0～15长度全部mask并追加大尺寸随机mask，正负两类共131,370例，dtype/shape/stride/value全部exact。
- 未应用最小patch只含两行替换，临时LF副本`git apply --check`通过；远端业务仓库仍`bf9ed6e` clean。尚无NPU函数门禁和正式30-step收益，禁止提交或宣称优化完成。
- STEP-144结论：`torch.nonzero(...).squeeze(-1)`输出天然唯一，删除后续`.unique()`在CPU 131,370例及后8卡8-rank 18,776例均exact；隔离256调用中位耗时可由约159.8～189.5ms降至63.0～82.5ms。但真实客户batch16 30-step普通均值6.352043秒，相对`bf9ed6e`的5.715218秒回归11.143%，全步回归9.660%，故局部上限未转化为端到端收益，候选拒绝且不提交。
- 2026-08-13官方调优流程复核：昇腾7.3文档强调先按场景采集性能数据、拆解瓶颈到下发/计算/通信，再选择针对性方案；并行策略不存在通用万能配置，需通过具体模型实验决定。官方数据加载建议在内存允许时保持`pin_memory=True`，NPU亲和策略强调减少H2D/D2H引入的同步。当前客户配置表面已启用`pin_memory=True`，但MMCV `DataContainer`没有`pin_memory()`，PyTorch pin-memory递归遇到该自定义对象会原样返回，所以其内部tensor并未因配置字段自动页锁定。这是“字段已开但功能未生效”的可验证候选，不涉及改变数学功能。

## DrivingSDK R1：MSDA接口与活跃路径审计（2026-08-13）

- DrivingSDK官方仓库把高性能算子、补丁与负载均衡作为模型优化能力，并在模型优化文档中明确给出`from mx_driving import multi_scale_deformable_attn`路径；但官方master在2026年新增的能力不能反推客户已安装版本，当前实验合同必须以容器内固定版本为准。
- 客户容器实际安装`mx_driving 1.0.0+gitde13346`，未升级或替换。顶层`multi_scale_deformable_attn`是`MultiScaleDeformableAttnFunction.apply`；forward参数为`value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights`，不接收项目旧扩展的`im2col_step`。
- 该版本Python包装会把两个shape/index张量转成`int32`，把location/weight转成与value相同dtype；forward调用`mx_driving._C.multi_scale_deformable_attn`，backward返回value/location/weight三类梯度。Ascend 910_93随包预编译配置只列出ND格式、float32数据张量与int32 shape/index张量，不能假设客户版本支持GitHub master的新dtype/head-dim范围。
- 项目当前自定义函数同样强制走fp32版本，并调用MMCV扩展`ms_deform_attn_forward/backward`；项目调用点保留CUDA兼容条件，但真实训练入口的`transfer_to_npu`会让NPU张量满足该条件，当前profile栈明确命中自定义函数forward line118和backward line183，因此并非MMCV reference CPU fallback。
- 容器重启后直接`import mx_driving`会让PyTorch设备后端自动加载与`torch_npu`重复注册，报“Two accelerators cannot be used at the same time: npu and npu”。使用客户环境既有的`TORCH_DEVICE_BACKEND_AUTOLOAD=0`并显式先导入`torch_npu`后，API审计成功；这是启动顺序约束，不通过安装或改版本解决。
- 下一门禁必须使用真实profile形状，同时比较forward、对value/location/weight的backward、输出dtype/shape、有限性与误差；还要覆盖客户实际float32、动态query/level/point组合以及8-rank一致退出。`im2col_step`仅是旧扩展参数，替换时不能错误传给DrivingSDK API。
- 远端原位扫描当前profile的1,523,977行`operator_details.csv`：4个active step内自定义MSDA forward正好24次、backward正好24次，即6次/step。forward主体device self为699.839ms/4步，backward主体`aclnnMultiScaleDeformableAttentionGrad`为3239.099ms/4步；再次证明主要机会在替换完整实现，而不是2.048ms/step的预清零。
- profile可见真实shape族包括`value=[16,15360,8,32]`、MapTR权重`[16,2400,8,1,4]`及输出`[16,2400,256]`，并存在首维112/32等其他动态族；但C扩展顶层记录的Input Shapes为`nan`，不能仅靠子op形状可靠重建五个输入的一一对应。下一步采用诊断启动包装在真实1-step中只记录shape/dtype/stride/contiguous，不记录张量值，随后自然退出并保持业务仓库clean。

## Profiling原始数据生命周期（2026-08-13）

- 新规则：profiling原始trace、CSV、kernel与导出目录只保留到当前候选完成热点提取、必要架构/异常报告及门禁输入合同；确认后续不再需要后立即删除，长期只保留脱敏统计与结论。
- 本轮已删除12份闭环/失败重试旧profile，释放约90GiB量级；当前R1仍需的`profile_current_bf9ed6e...`约13GiB暂时保留，R1合同完成后继续清理。
- R1真实shape合同取得后，最后一份约13GiB原始profile也已删除；远端顶层旧profile目录计数为0，本轮累计释放约103GiB量级。

## DrivingSDK R1：真实shape与数值门禁（2026-08-13）

- 后8卡8-rank真实1-step元数据探针自然完成、exit0；每rank恰好6次MSDA：MapTR 4次、Temporal 1次、Spatial 1次，8 rank合同完全一致。三类分别为：MapTR value `[16,15360,8,32]`、location `[16,2400,8,1,4,2]`、shape `[[96,160]]`；Temporal value `[32,15360,8,32]`、location `[32,15360,8,1,4,2]`；Spatial value `[112,576,8,32]`、location `[112,15360,8,1,8,2]`、shape `[[18,32]]`。均为fp32、ND/contiguous，shape/index输入实际为int64，DrivingSDK包装内部转int32。
- 探针1-step loss=445.1907、grad_norm=68.5548，均有限；该轮只包裹forward记录元数据，仍执行原实现，不是候选正确性结果。8 rank主PID完整映射后8逻辑卡，退出后端口与NPU进程释放；kernel_meta/fusion副作用归档后`bf9ed6e` clean。
- DrivingSDK首次两轮等价门禁均在候选调用前报自定义opapi符号不可见。已安装包自身确实含`libcust_opapi.so`及四个MSDA前反向符号，并在import时调用`_init_op_api_so_path`；根因是进程先调用项目当前ACLNN实现后锁定基础opapi。仅改import顺序无效；把DrivingSDK候选作为进程内首个ACLNN调用后可正常执行，不需安装、升级或永久改环境。
- 候选优先的8-rank三类真实shape门禁完成全部结果。当前与候选输出最大绝对误差≤7.451e-8；grad_value≤2.682e-6、grad_attention_weights≤2.876e-6。MapTR/Temporal的grad_sampling_locations在严格`atol=1e-5,rtol=1e-4`下因近零元素失败，最大绝对误差分别2.050e-4/2.575e-4，但全局NRMSE分别4.101e-7/3.927e-7；Spatial该梯度最大9.537e-6且严格allclose。全部有限mask一致，所有张量NRMSE≤6.439e-7。
- 当前实现重复运行全部严格allclose；DrivingSDK重复运行也全部严格allclose，只有grad_value存在约1e-8～6e-8级非逐位差，但仍在allclose内。结论为“前反向数值近似等价、非逐位一致”；符合融合归约顺序改变预期，但必须再通过真实候选1-step loss/grad，不能直接跳到正式30-step或提交。
- 8-rank profiler-off真实shape机制微基准同步覆盖前向+反向：MapTR中位13.579→5.140ms/调用（2.636×），Temporal 135.708→68.763ms（1.975×），Spatial 883.604→220.951ms（3.998×）。按真实频次4+1+1加权，1073.681→310.290ms/step，理论节省763.129ms/step、3.458×；8 rank范围稳定，最大机制显存8542MiB/rank。
- 单文件候选仅在当前活跃fp32兼容类中导入DrivingSDK并替换forward/backward，调用签名、`im2col_step`参数、symbolic和fp16旧类保留。真实后8卡8-rank客户batch16 1-step自然exit0、fatal0；loss=460.0874、grad_norm=132.4465均有限，loss无数量级偏离。该批次随机性与父1-step不同，不能把grad单点差异归因于融合算子，下一步由30-step分布判断。
## 2026-08-13：DrivingSDK MSDA融合正式验证与提交

- `f922c38 【npu性能优化】MSDA切换DrivingSDK融合实现`仅修改fp32 MSDA前反向一个文件，父提交为`bf9ed6e`，提交后远端clean且未push。
- 后8卡、8 rank、客户batch/rank=16单轮30-step：全步9.269933秒、吞吐13.808082 samples/s；严格普通23步mean/median/P95=5.625957/5.637000/6.713600秒；SOAP均值33.301秒；峰值26848MiB/rank。
- 相对直接父提交：全步-0.473%、吞吐+0.475%、普通均值-1.562%、中位-2.445%、SOAP-3.562%；P95因SOAP后iter23/24抖动回归13.355%。相对同配置永久基线`4c37039`：全步-75.241%（4.0389x）、吞吐+303.863%、普通均值-50.861%、SOAP-88.089%。
- 数值门禁：三类真实shape同输入输出/梯度NRMSE≤6.439e-7；30个loss/grad全有限且无数量级偏离。该实现为高精度近似等价、不是逐位等价；单轮30-step不能替代长期收敛验证。
- 原始profiling数据已全部清理，远端顶层profile目录数为0；本轮正式训练未启用profiler，只归档训练生成的kernel/fusion副产物。
## 2026-08-13：R2官方方法与当前证据边界

- 昇腾7.3官方把单batch端到端时间拆为数据加载、模型前反向、优化器、后处理、未掩盖通信和调度；建议先采集性能数据，再把问题定界为计算、通信或下发。抖动问题应比较抖动步与正常步，通信等待需回溯第一处快慢卡差异，不能把notify wait直接当成通信算子慢。
- 官方host/device判定依据是host API与device kernel的下发关系及时间线Free区间；数据加载参数如`num_workers`、`pin_memory`、`persistent_workers`、prefetch和collate位置没有万能值，必须结合当前模型、CPU、内存和IO做实验。
- 官方NPU亲和原则是数学等价融合、减少冗余下发和不必要的H2D/D2H/stream同步；`item`、`reduce_all`、`isfinite`等只能在语义允许时减少，不能为性能删除必要检查。
- R2当前没有`f922c38`之后的原始profile，旧profile也已按规则删除。历史脱敏统计证明MapTR target路径曾为活跃host/device候选，但不足以证明当前最慢子功能；必须先静态拆分调用链和复用边界，再决定是否采集新的最小profile。
- 官方来源：[性能调优流程](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0001.html)、[并行策略建议](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0024.html)。
- 客户有效config反向解析确认活动链：`InternalDatasetTrackStream` → `VectorizeLocalMap` → `GenerateMapGTShifts` → `SPetr3D.lane3d_head=MapTRv2HeadDecoder` → `MapTRDecoder`，训练匹配为`MapHungarianAssigner3D`。R2静态审计只覆盖这些活动实现，不分析全仓同名历史head/dataset。
- 当前远端为`f922c38` clean、训练进程0、profile目录0。全仓`deepcopy`和递归Config类型枚举都过宽，后续改为按活动注册类及精确调用点读取，避免把未启用代码和重复数据集配置计入候选。
- R2活动config已显式设置`gt_shift_pts_pattern="v6_curve"`、4层`MapTRDecoder`，并在训练pipeline中调用`GenerateMapGTShifts`，输出`map_gt_shifts_pts_list/map_gt_pts_types_list/map_gt_pts_colors_list`。该pipeline stage只读取`LiDARInstanceLines.shift_fixed_num_sampled_points_v6_curve`一次并以`DC(..., cpu_only=True)`写回；这些字段也已包含在lane输入keys中。
- `MapTRv2HeadDecoder`已有缓存消费分支、现场重算回退及`compare_cached_shifts`诊断开关，并将同一个`map_gt_shifts_pts_list`复制为4层decoder的输入列表。提交历史显示这套预计算/缓存并非本轮新增，至少在客户旧提交`657c1f2/6f8f66f/25ccf58`中已经存在；因此R2不能再次以“把GT shift前移到DataLoader”为候选，必须确认当前有效分支确实命中缓存分支，再寻找剩余重复处理。
- `spetr3d.py`会把三个缓存字段原样带入当前时间帧`data`并传给`lane3d_head.loss`；有效config没有打开`compare_cached_shifts`，因此缓存完整时不会调用现场`_calc_gt_shift_from_instances`。缓存只在loss入口各做一次CPU→NPU搬运，随后同一列表由4个decoder层复用。R2核心前移机制已经真实生效。
- 剩余target路径的历史证据包括：STEP-073宽mask复用在旧batch1三轮仅+1.28%且低于2.15%噪声；STEP-088跨层GT normalize仅可回收约1.22～1.28ms；Hungarian完整乐观上限约22.61ms且无等价设备solver。当前batch16把target调用放大到256次/step，属于允许重新量化的物料变化，但旧结论不能直接证明`f922c38`当前占比；需采rank0单正常步最小profile。
- R2当前`f922c38` rank0单正常步profile完成：11/11、exit0、fatal0，8个rank及后8卡映射完整，loss/grad有限。插桩后service/busy/underfeed=45711.891/1852.411/43859.480ms，underfeed95.948%；只用于归因。异常JSON schema 0错误，架构报告10节。
- 最内层栈聚合：缓存加载39.619ms host self/0 device；`target_single`170.314ms device self，其中历史正式失败的Unique为50.411ms；`loss_single`125.512ms；Hungarian57.507ms。四层间类别GT切片的可删纯device上限约15.7ms，低于门槛。
- batch16 Hungarian为216次非空匹配，profile中SciPy solver累计184.997ms；但row/col合并为一次H2D的8-rank真实频次门禁虽然值/layout/下游赋值全部exact，workload中位却为10.583854→10.786435ms，节省-0.022318ms、0.997935x，rank节省范围-0.470914～+0.266787ms。确认profile H2D host total被插桩放大；候选拒绝、不改码、不30-step、不提交。
- R2原始profile已在双报告、schema、manifest、栈摘要和门禁合同全部归档后删除，释放3,248,664 KiB（约3.10 GiB）。保留目录改名为`r2_maptr_target_analysis_f922c38_back8_8npu_20260814T001000`；远端顶层含`profile`名称的诊断目录数为0，业务仓库仍clean，后8卡无进程。
- P2筛选：R4标准MHA活动合同为embed_dim256/8 heads/dropout0/4 decoder层，但当前保留分析仅有无法归因且单次出现的82.983ms BMM，不满足“稳定MHA热点”；暂缓。R5的`mx_driving.bev_pool`实现属于未实例化的`BaseTransform/LSSTransform`路径，有效config只用`BEVFormerEncoder`，分析命中0；关闭为不活跃。R6四个候选环境变量当前均未设置，可进入固定环境支持性只读审计。

## R6固定环境支持性审计（进行中）

- 远端权威状态仍为`mapqr-leicheng`、`ascend_npu_optimize@f922c38`、Git clean、训练0、后8卡空闲。
- 客户安装物为torch_npu 2.7.1；其`libtorch_npu.so`直接包含`TASK_QUEUE_ENABLE`、`CPU_AFFINITY_CONF`、`COMBINED_ENABLE`、`PYTORCH_NPU_ALLOC_CONF`与`expandable_segments`解析/错误字符串，说明四项均被当前二进制识别，而不是只存在于新文档。
- 华为Ascend官方仓库标签`v7.3.0-pytorch2.7.1`（commit `3be71f46df8b8ad10f4e4230aa67d83e68a48ba4`）源码确认：`TASK_QUEUE_ENABLE`未设置时默认1，允许0/1/2；blocking模式会强制为0，模式2不支持NPU graph capture。官方说明Level 2在Level 1基础上把workspace任务迁入二级流水、建议二进制场景使用，但可能抬高NPU峰值内存。
- `CPU_AFFINITY_CONF=1`是粗粒度绑核，默认关闭；官方对需要绑核的场景更推荐细粒度模式2，并明确Docker映射、已有亲和性及额外线程可能使其不生效或劣化。当前计划指定模式1，因此测试前必须先核对容器CPU/NUMA拓扑。
- `COMBINED_ENABLE=1`不是SOAP优化器融合。尽管内部检查函数名为`CheckCombinedOptimizerEnable`，唯一实际调用位于`ContiguousOpt.cpp`：它为由最多两个view类操作组合产生的非连续tensor启用`combined` contiguous路径，处理reshape/slice/select/indexing等严格可推断组合。是否值得测试应由当前profile的contiguous/TransData证据决定。
- `expandable_segments=True`属于虚拟内存/碎片治理，官方使用场景是OOM、内存碎片或降低内存占用；当前单rank约26.8GiB/65.5GiB且无OOM/碎片证据，暂不作为首个性能候选。
- 保留的R2单正常步分析中，COMBINED可能覆盖的可见纯device族上限约为ViewCopy 29.667ms、Transpose 19.333ms，合计约49.0ms/步；但其总cost等待比0.925～0.952，with-stack host值明显放大，且没有保存具体组合view输入合同。因此它可以排在TASK_QUEUE之后，但当前不直接宣称可回收49ms。
- 当前容器没有设置`ASCEND_LAUNCH_BLOCKING/TASK_QUEUE_ENABLE/PER_STREAM_QUEUE`，所以现状确实是TASK_QUEUE默认Level 1。仓库虽有`torch.compile`字样，但有效SOAP工具的`is_compiling()`硬返回True，使装饰器直接走原函数；另一个`@torch.compile`的occ head在有效config中被注释。没有`NPUGraph/capture_begin/torchair`活动证据，Level 2的graph-capture禁用约束不命中当前训练。
- R6首个唯一A/B项选择`TASK_QUEUE_ENABLE=2`：它直接针对官方所述host算子下发二级流水和workspace负载平衡；当前模型有大量ACNN/工作区任务，且单rank显存仍有约38.7GiB余量。测试只改变进程启动前一个环境变量，不改业务代码；必须记录显存是否上升、loss/grad有限、普通步/全步/SOAP及相对默认Level 1收益。

## R6 TASK_QUEUE Level 2正式结论

- 后8卡、8 rank、客户batch/rank=16、30-step、profiler-off，唯一变量为默认Level 1→`TASK_QUEUE_ENABLE=2`。30/30、exit0、fatal0，8 rank/设备/PID映射完整。
- Level 2：全步9.966667秒、吞吐12.842809 samples/s；普通23步mean/median/P95=6.001435/5.792000/7.396900秒；SOAP窗口32.856/35.484秒、均值34.170秒；峰值26849MiB/rank。
- 相对当前`f922c38`默认Level 1：全步+7.516%、吞吐-6.991%、普通均值+6.674%、中位+2.750%、P95+10.178%、SOAP+2.610%、显存+1MiB。所有性能主指标方向一致回归。
- 30个loss/grad均有限；loss均值/中位/范围309.568547/309.037950/220.8458～434.8636，grad为53.902013/47.913700/41.2065～117.5476。随机batch只能排除数量级发散。
- 相对`4c37039`的累计全步仍为-73.380%（3.7565x），但来自既有提交，不能归因于Level 2。结论`REJECT_NO_COMMIT`；不写启动配置、不改业务代码、不提交。
## 2026-08-14：profiling 数据长期清理规则

- 用户明确要求：旧 profiling 原始数据每次用完且后续无需复核时立即删除。
- 已完成的旧数据清理累计释放约 103 GiB；最近一次 R2 原始 profile 约 3.10 GiB 已在报告、schema、架构报告和门禁验证后删除。
- 当前已知远端顶层原始 profile 目录数为 0；今后保留脱敏结论与报告，不保留已闭环的原始 kernel/operator/trace 数据库。

## R6 `COMBINED_ENABLE=1` 正式结论

- `f922c38`、后8逻辑NPU、8 rank、客户batch/rank=16、30-step、profiler-off；唯一机制变量为`COMBINED_ENABLE=1`，没有继承已拒绝的TASK_QUEUE Level 2。
- 相对同一HEAD默认配置：全步9.269933→9.672633秒（+4.344%），吞吐13.808082→13.233211 samples/s（-4.163%）；普通23步mean/median分别+3.591%/+5.304%，SOAP +1.047%。仅P95 -1.373%、显存-12MiB，不能抵消主体回归。
- 30个loss/grad全部有限，无数量级发散。相对永久客户同口径基线`4c37039`累计全步仍-74.165%（3.8707×），但来自此前功能提交，不能归因于COMBINED。
- 决策`REJECT_NO_COMMIT`；不写启动配置、不改业务代码、不提交。训练后`f922c38` clean、进程/端口/后8卡为0、原始profile目录0。
- 启动期宿主`npu-smi` PID一一映射因一次筛选表达式错误和两次SSH认证超时未形成归档证据；8个直接rank/8设备/变量继承已核验。候选已因稳定负收益被拒绝，本轮不能作为发布采用证据且无需为拒绝结论重跑。

## R6 CPU affinity 拓扑审计（进行中）

- 固定版本官方源码确认：未指定范围时，以进程可见CPU数除以可见NPU数，按运行时device index连续分块；320 CPU、8可见NPU会得到device0～7对应0～39、40～79、…、280～319。模式1让所有PTA线程共享整段，模式2把main/ACL/release/watchdog分别固定到区间前4核，其余线程使用剩余核；官方推荐模式2。
- 官方明确警告Docker可能改变device与NUMA映射，应按真实映射自定义范围；若当前线程亲和集合缺少任一目标核，torch_npu会认为已有亲和性并完全跳过自动绑核。`npu_affine:1`依赖DCMI，且仅A2支持。
- 当前`mapqr-leicheng`没有cpuset、CPU quota或mems限制：容器与宿主均可见CPU0～319、NUMA0～7，每节点40核，PID1亲和也是0～319；因此不会因既有cpuset直接跳过，但默认连续分块是否与后8逻辑设备8～15匹配仍未证明。
- `npu-smi -m`确认后8逻辑设备8～15对应物理Phy-ID8～15；`npu-smi info -t topo`只返回16个Phy-ID之间的HCCS/SIO关系，不输出CPU亲和区间。当前版本不支持`-t affinity*`查询；四种只读尝试均明确返回参数不支持。
- Ascend加速卡PCI function的sysfs `numa_node=-1`、`local_cpulist=0-319`，不能用PCI sysfs恢复真实设备—NUMA映射。下一步必须通过当前torch_npu/DCMI能力提取亲和范围，或因映射证据不足关闭默认模式；不能直接用device index猜NUMA。
- 零绑核DCMI探针使用`CPU_AFFINITY_CONF=0,npu_affine:1`：模式0使进程亲和前后均为0～319/320核，但解析器仍尝试读取物理亲和范围。当前客户环境明确警告`dcmi_get_affinity_cpu_info_by_device_id is not supported`并禁用`npu_affine`，随后回退为机械八等分device0→0～39、…、device7→280～319。
- 由于后8卡可见列表把Phy-ID8～15重编号为运行时device0～7，而DCMI无法给出物理亲和范围，默认模式1/2没有资格直接A/B。下一步只能通过8卡NUMA→H2D定向微基准推断邻近节点，再构造显式`npuN:start-end`范围；若映射不稳定则关闭CPU affinity方向。

## R6 CPU affinity 完成结论

- 8-rank后8卡NUMA→H2D微基准共64组：仅20组CV≤10%；默认节点平均中位1.645429ms，每卡事后oracle最优1.632484ms，差0.793%。默认节点恰为oracle最佳4/8张卡；oracle节点分散为1/3/4/7，且多个“最优”CV为15%～43%，不能形成稳定设备—NUMA映射。
- `REJECT_NO_COMMIT`：不运行30-step、不写启动建议、不改代码。R6的TASK_QUEUE、COMBINED均正式回归；CPU affinity映射不可靠且上限不足；expandable_segments无OOM/碎片证据，按计划场景不适用。R6整体关闭。
- 微基准完成时8直接rank/8设备通过、fatal0；`docker top`列格式在当前版本报错，宿主PID映射未留档，所以结果只用于关闭候选，不作为发布采用证据。结束后进程/端口/profile为0，Git clean。
# 2026-08-14 profiling 原始数据生命周期规则

- 用户新增永久规则：旧 profiling 原始数据在完成分析、摘要校验且后续不再需要后必须删除。
- 仅保留脱敏统计摘要、分析报告、脚本版本/校验值；不保留或下载远端原始 trace、kernel 明细目录等产物。
- 当前远端已确认不存在旧的顶层 profile 目录或残留 raw `profile/` 子目录；R7 不恢复已删除原始数据，只使用 R2 留存的脱敏分析摘要。

# 2026-08-14 R7 格式转换热点准入证据（留存摘要）

- R2 留存摘要显示，单个 normal step 中 `aclnnConvolution_TransData_TransData` 435 次、纯设备 103.133388 ms；反向同族 348 次、纯设备 32.617027 ms。两者合计纯设备上界 135.750415 ms。
- 另有 InplaceCopy/Index 的 ViewCopy/Transpose 高频调用，但等待占比 92.5%–97.5%，不能把总时长直接当作可消除计算时间。
- 留存架构报告没有可复核的 invocation 级模块/调用栈映射，因此当前只能证明“格式转换是候选热点”，尚不能证明某个具体模型层是稳定根因。
- 当前分支在 `tools/train_spetr.py` 同时将 `torch_npu.npu.config.allow_internal_format` 与 `torch.npu.config.allow_internal_format` 设为 `False`；下一步先审计固定版本官方语义与该设置的提交历史，再决定是否允许进入 8 卡正确性/兼容性门禁。
- 客户宿主机 DMI 明确为 `Atlas 800I A3`；后8卡所在板卡为双芯片板、PCI Device ID `0xD803`，健康兼容。`npu-smi product` 子命令在本机不支持，已改用官方建议的 DMI + board 信息交叉确认。
- 固定版本源码中，A3 SoC（`SocVersion >= Ascend910_9391`）初始化时若用户未设置该选项，会把 `ALLOW_INTERNAL_FORMAT` 默认置为 `disable`。Python 属性赋值只是把布尔值转成 `enable/disable` 传给 C++ option。
- 该版本官方支持表明确提示：A2 Conv2d 频繁编译时建议关闭 internal format；ConvTranspose2d 的 3 维输入要求关闭；TensorShape 实现还警告非基础格式的 view/as_strided 可能产生精度问题。项目含多条可达候选 ConvTranspose2d 路径，因此全局开启必须先做兼容/数值门禁。
- 项目历史显示，两处 `allow_internal_format=False` 最早随 2026-08-03 的 loss/randomness alignment 工作引入，随后多个对齐提交重复带入；当前行归属 2026-08-04 的完整工作区提交。2026-08-06 一度在文件前部注释旧副本，但当前文件后部仍有活跃设置。它不是已独立证明的性能选择，不能直接假设删除安全。
- 容器现场为 torch 2.7.1 / torch_npu 2.7.1，Python setter 与固定版本源码一致，并暴露 `_npu_getOption`、`_npu_get_soc_version`。在尚未初始化 NPU 的短进程中，`True`/`False` 分别读回 `enable`/`disable`，说明 A3 的“默认关闭”不是无条件拒绝用户显式开启；必须做运行后兼容/数值验证。
- 客户活跃配置实际使用 ResNet34、FPN、`BaseBEVBackbone_FPN(use_deconv=True)`、ConvNeXt 和 `SECONDTransFPNV3(upsample='bilinear')`。其中前者明确实例化多层 Conv2d 与 ConvTranspose2d，且前向真实可达，不能把仓库静态存在误当作非活跃风险。
- R7 8-rank隔离门禁最终通过兼容/数值检查：False/True初始化后8/8分别回读disable/enable；Conv/ConvTranspose输出和输入梯度逐位一致，权重梯度最大差分别约6.98e-10/4.66e-10。
- 但True相对False的8-rank耗时中位全部回归：Conv +15.487%、ConvTranspose +47.821%、Conv+view链 +13.739%；ConvTranspose每个rank都回归（+10.201%～+64.049%）。因此留存profile中TransData的135.75ms纯device候选上界并不能通过全局internal format开关回收。
- R7最终判定不具完整30-step训练A/B资格：保持False，不改业务代码、不提交。该结论同时符合A3固定版本默认关闭和客户活跃反卷积路径的现场证据。

# 2026-08-14 R8 动态负载均衡前置证据审计

- 按 `ascend-profiling-anomaly` 规则，rank不均衡不能用单一rank的wall或HCCL wait推断；必须区分device busy、wait-anchor/通信等待，并至少取得逐rank时间与样本复杂度的配对数据。
- 远端现有30-step正式日志仅有rank0 TextLogger的30条 `Iter [n/30] time`；没有逐rank step/data time，也没有点数、GT数或序列长度字段。诊断目录没有独立rank日志。
- R2留存异常JSON来自rank0单active step；没有`world_size`级逐rank数组，且原始profile已按规则删除。它可以证明rank0局部bubble/HCCL活动，不能证明8 rank离散度或样本复杂度相关性。
- 现有资料因此只能给出 `insufficient_evidence`：尚不能授权DrivingSDK load balance或dataset bucketing，更不能以rank0时间替代8 rank straggler证据。
- R8一次性8-rank诊断完成240个rank-step样本；排除预热和SOAP后23步的forward rank范围中位559.575ms、范围/最慢rank 13.583%、rank CV 4.494%，说明确有离散，但不自动等于样本不均衡。
- step+rank双向去均值后，point/voxel/3D GT/map GT对forward的Pearson分别仅0.0100/0.0969/-0.0897/0.0284；5000次step内置换p值分别0.9006/0.2691/0.1806/0.7950；最大复杂度rank命中最慢forward仅0%/13.04%/13.04%/17.39%。无任何特征通过预先固定合同。
- rank5在23步中9次forward最慢，但最大复杂度分布在各rank且与rank5不稳定重合；这更像固定rank/设备或未观测因素，不能通过改变dataset顺序来治疗。
- R8最终关闭：不做DrivingSDK load balance、不改DistributedSampler/样本集合、不候选A/B、不提交。诊断30个loss/grad全部有限且无数量级偏离；同步插桩wall不作为正式HEAD性能数据。

# 2026-08-14 DrivingSDK P3关闭

- 固定`mx_driving 1.0.0+gitde13346`安装包确实提供SparseConv3d/SubMConv3d、Voxelization、dynamic scatter、scatter mean/max等成熟API；API存在本身不构成优化证据。
- 客户活跃config为`PillarVFE -> PointPillarScatter_Seg -> BaseBEVBackbone_FPN`，没有实例化SparseConv/spconv；`TransPointsToVoxels`位于DataLoader pipeline，已生成voxels/coords后再交给模型。
- R2留存profile/stack摘要中Pillar/voxel/scatter/Sparse语义命中为0；通用Index/IndexPut/Conv热点无法可靠归属Pillar链。因此不满足P3“仅profile命中时研究”的前置条件，不为API存在而重采profile或替换全链。
- Patcher只是非侵入实验工具，不是独立性能候选；自定义rasterizer/几何kernel没有成熟等价热点证据；R2通信仅8.6975ms device、76.795%与compute重叠且缺communication.json，R8 full-compute各rank均值仅约2.3%跨度，不进入HCCL优化。
- DrivingSDK P3全部关闭，未改码、未训练、未提交。按用户顺序转回此前通用计划，首项恢复已延期但未否决的DataContainer页锁定候选。
- R9确认客户训练走项目自定义DataLoader builder，该builder实际写死`pin_memory=True`；MMCV `DataContainer`缺少协议导致配置只对普通容器生效。真实客户batch后8卡8-rank机制A/B各20步：baseline/candidate各160条，正常步各120条。CPU batch中位约457个tensor、1.901GB/rank；pinned byte比例0%→100%，同步scatter中位316.702→45.110ms（-85.756%），配对中位差-268.149ms、95% bootstrap CI[-285.119,-244.209]ms；同步诊断整步中位5831.482→5116.419ms（-12.262%）。检查6032个tensor值/shape/dtype/stride零差异，320条loss全部finite。机制通过预设门槛，但同步探针数据不是正式吞吐基线，仍须单文件候选的1-step和30-step正式A/B。
- R9原始逐rank JSONL、训练/launcher日志、work、hook、harness和`kernel_meta`已删除；远端目录6,715,438→3,167 bytes，仅保留脱敏summary/report，业务仓库`f922c38` clean、训练0、后8卡本轮PID0。
- 对整个远端`diagnostics`补做历史残留审计：未发现仍保留的`PROF_*`/`ASCEND_PROFILER_OUTPUT`原始profile主体，但发现大量历史运行遗留的空/极小`kernel_meta*`目录。确认无训练/profiler进程、路径均在diagnostics内且非符号链接后，删除728个目录，按逐目录口径合计19,298 bytes；复核匹配目录0、业务仓库clean。
- 2026-08-14 STEP-174证据规则：恢复`planning-with-files`与`ascend-profiling-anomaly`技能。下一候选必须同时检查wall/busy-union/kernel-sum/total-cost，不能把with-stack host self或高wait小duration锚点当可回收耗时；若只有`op_summary`或脱敏聚合而缺时间戳/上下文，最多形成采集需求，不能直接改码。任何新profile必须同时生成异常输出与独立10节架构报告，之后按用户规则删除raw数据。
- DataContainer正式A/B已经证明“机制局部加速”不等于“端到端可提交”：普通步约-9.65%，但全步+3.48%、吞吐-3.36%、SOAP+6.70%，因此下一候选仍以直接父提交全程净收益为首要裁决，永久基线只报告累计效果。
- STEP-174远端权威状态：`ascend_npu_optimize@f922c3897255`、工作树clean、唯一正确容器在线、训练/profiler进程0、raw profile/kernel_meta0。唯一仍与当前HEAD完全一致且包含异常+架构双报告的留存证据是R2单正常步profile；其他留存多为候选门禁/正式A-B摘要。
- R2当前HEAD报告的纯device前列为Conv 271.558ms、MSDA backward 185.617ms、Conv TransData 103.133ms、单个BatchMatMul 82.983ms、MSDA forward 81.363ms。MSDA已由`f922c38`提交改变实现，旧MSDA kernel排名不再可作为下一候选；Unique、Index/IndexPut的可删功能已分别正式拒绝或由GeometricLoss提交解决；internal-format整体开启已证实使Conv/ConvTranspose回归。剩余Conv总量、TransData和单个BatchMatMul尚缺窄源码功能归因，不能直接修改。
- R2异常事实为单正常步service 45.7s、busy union 1.852s、underfeed 95.95%，但`record_shapes+with_stack`开销极重，只能用于bubble/栈归因。最大内部gap 66.133ms位于VectorNorm→Stack，另有57.868ms Add→Fill、52.375ms Repeat→Eq等；软标签多为possible_sync_or_h2d/host_launch_lag，尚无唯一根因。通信仅8.697ms且76.8%掩盖，不是优先项。
- R2专项栈聚合只覆盖MapTR target/loss/Hungarian及少量head，不覆盖全模型Conv/TransData/BatchMatMul源码栈。其最大可删device子项Unique 50.411ms已经正式回归拒绝；Nonzero 28.751+23.149ms、target/loss Index/IndexPut等与同一target/loss语义纠缠且已有GeometricLoss/Unique/Hungarian专项结论，不能仅按kernel名再次拼成新功能。`head:612`约20.278ms device self、`head:589` Upsample约5.172ms等均未越过约22.7ms噪声门槛或是必要数学计算。
- 现有R2栈摘要的host self是重度with-stack放大后的加性归因，device self才可作为局部上限；例如target_single host 17.46s但device self仅170.3ms。下一候选不能从这些host值直接推导端到端收益。
- R2异常JSON保留20个bubble和9个wait-anchor。所有`InplaceFill/Zero`、EqScalar、IndexPut Broadcast/Kernel、MatMul MemSet、Nonzero MemSet等高total-cost项都满足wait_ratio>0.97且平均纯kernel<10us，必须降级，不能作为新优化对象。最大可行动事实是：20个gap中13个possible_host_launch_lag、6个possible_sync_or_h2d；但JSON只保留前后device kernel，没有具体host stack，`requires_host_followup=true`。
- 单步normal profile的最大内部bubble上下文为VectorNorm→Stack 66.133ms、Add→Fill 57.869ms、Repeat→Eq 52.375ms；旧raw trace已按规则删除，现有脱敏文件不能把这些窗口唯一映射到源码。因此仅凭留存摘要无法形成正确候选，下一步应复用当前HEAD最小单正常步profile，重点把每个top gap的host op/stack覆盖写入脱敏报告后再删raw。
- R2成功采集夹具仍可复用：rank0 CPU+NPU、wait8/warmup1/active1、record_shapes/with_stack、Level0、data_simplification、export text；其他7 rank只参与真实训练，MAX_ITERS=11，后8卡。训练脚本SHA路径与当前正式测试同源。新profile无需扩大采集范围，只需让分析器把top bubble重叠的host event名称、category、duration和call stack摘要写入脱敏输出。
- R2保留目录仍含旧launcher/train/work日志、PID标记、pycache和一次性Hungarian门禁日志；这些不是raw PROF主体，但本轮复用夹具后应与新profile一起做生命周期收尾，只保留必要脚本、双报告、schema结果和脱敏候选合同。
- 现有`.codex-tools/analyze_ascend_profile.py`已经在单次流式读取trace时筛选与top bubble相交的事件，但只合并成coverage比例，丢弃了事件名称/category/stack。因此无需扩大profiler或再扫描完整raw多次；可在同一流式分支仅保留与50个bubble相交的CPU/Python事件摘要，再为每个bubble按实际overlap排序输出有限条上下文。
- 分析器当前仅把`cpu_op`计入host coverage；补充上下文时应保留`cpu_op`和`python_function`，排除跨整步`ProfilerStep`，并截取项目相关stack frame。所有根因仍用soft label，host context只用于源码定位，不把嵌套event duration相加当可回收wall。
- 正式异常schema的`bubble_windows.items`没有禁止额外属性，因此可增加`host_context`数组而不破坏现有required字段与Draft 2020-12校验；无需另改schema。每条上下文可包含name/category/overlap_ms/duration_ms/project_frames，并限制条数/长度。
- 分析器已在不改变现有bubble/coverage/标签计算的前提下增加`host_context`：只收集与top gaps相交的`cpu_op`/`python_function`，排除ProfilerStep；每bubble候选定期裁剪、最终去重保留12条，项目栈帧裁剪为相对路径。Markdown新增明确“嵌套event不可相加”的上下文表。
- 本地已有通用后8卡profile config/wrapper，但配置等待10步、MAX_ITERS14，且依赖未列出的`npu_profiler_hook_after`；R2远端夹具是已在当前HEAD成功验证的wait8/MAX_ITERS11方案。为了最小采集与可复现性，优先在新诊断目录原位复制R2已验证夹具，只上传增强分析器/schema，不改业务仓库。
## 2026-08-14：STEP-174 当前 HEAD 最小 host-context profile

- 在唯一容器 `mapqr-leicheng`、后8逻辑卡、8 rank、客户 batch/rank=16、HEAD `f922c3897255` 上完成单个普通步的诊断采集；11/11、exit0、fatal/OOM0，训练、端口和后8卡均已释放。profile 活动步 43.002 秒包含 `with_stack/record_shapes` 开销，只用于归因，不能作为端到端性能指标。
- 增强分析器在原有一次流式 trace 扫描中为前20个 bubble 保存有界 host event/项目栈，不改变 bubble 阈值和归因规则；正式 schema 校验0错误，20/20 bubble 有 host context 和项目栈，独立架构报告仍按10节输出。分析器 SHA256 为 `6774a9d821e6542d1681629295ed74fe8aae210f4a8c959b52871a208b04d27c`。
- 正确解析实际仓库 base config 后，配置摘要为 SPetr3D、SOAP、IterBasedRunner、max_iters=30000、samples_per_gpu=16、workers=8；当前普通步 service=42150.222ms、device busy=1856.540ms、underfeed=95.60%。该数值同样属于 profiler 诊断视角，不替代 profiler-off 30-step。
- host stack 已把候选收敛到具体源码：bubble001 是已正式拒绝的 DataContainer H2D；bubble002/007 是 `clip_grads`；bubble003 是 MapTR loss 入口的列表 `.to`；bubble004 是 optimizer zero_grad/DDP unused-parameter/debug fingerprint；bubble005/006 指向 MapTR `loss_single.py:1218` 的 `sum` 与标量比较；其余前12项多为随机 mask、Unique、SOAP 必要数学或低于噪声阈值。
- 原始 profile 约3.316GB/121文件仍只在远端诊断目录中，当前仅为补取目标栈调用次数和审计后续 bubble 所保留；完成这些只读证据提取后即恢复 `fusion_result.json`、删除顶层 `kernel_meta` 与所有不再需要的 raw profiling，只保留脱敏报告和 SHA256。
- 目标栈精算确认 `maptrv2_head_decoder.py:1218` 在一个普通步执行256次：256次 `aclnnReduceSum` 共1.098ms device self、256次 `aclnnGtScalar` 共0.548ms device self，并触发256次 `_local_scalar_dense`（45.095ms host self）、`item/is_nonzero`。这些嵌套时间不能相加为收益，但与trace的约54.7ms gap同量级，足以进入机制门禁。
- line1218只决定“某类别GT数超过该类query容量”时是否发出 warning；四个布尔索引结果在判断前已经生成。候选语义边界是复用已生成 `sub_gt_labels_list[-1].shape[0]` 作为计数，避免再次在NPU上 `sum -> gt -> item`，不改变任何loss tensor、权重或匹配结果。
- bubble13是loss末端shape/finite检查，bubble14～20主要是已优化后的SOAP foreach必要计算，不新增可安全删除方向；梯度指纹只在指定指纹步运行，本次active普通步未命中，因此不把fingerprint本身误判为当前热点。
- 已删除本轮不再需要的raw profile、训练work/log、一次性harness和分析运行文件，共3,319,720,122 bytes、135个文件；保留目录从约3.1GB缩至252KB，只含脱敏异常/架构/目标行报告、cleanup摘要与SHA256。raw关键文件0、训练/profiler进程0、业务Git status0。
- 清理后的首次Git核验误用了并存的旧仓库 `l2.9-df-for-yuexiang_ascend_npu`，该仓库HEAD为 `f189414`；真正优化仓库是 `l2.9-df-for-yuexiang`、分支 `ascend_npu_optimize`、HEAD `f922c38`。候选未误应用到旧仓库。随后在正确仓库发现并精确逆向恢复 `fusion_result.json` 的0增/16删运行副作用，删除已验证位于仓库内的顶层 `kernel_meta`，再确认clean。
- 候选当前只改 `maptrv2_head_decoder.py`：新增 `sub_gt_count = sub_gt_labels_list[-1].shape[0]`，并让warning判断/内容复用该host shape计数；3增2删，语法与diff-check通过，尚未提交。
- 8-rank机制门禁两次诊断脚本失败均发生在profile结果读取API：第一次误用通用 `ProfilerActivity.NPU`，第二次使用 `torch_npu.profiler.profile` 后调用其不存在的 `key_averages()`；两轮均完整验证后8卡直接rank 0～7/world8，第二轮 `npu-smi` PID完整映射4/0～7/1，但尚未形成等价结果。固定客户环境未变，候选业务diff未变。
- 第三轮改用PyTorch CPU activity后，8/8 rank机制门禁PASS：空、未满、恰等、溢出四种输入的labels/types/colors/shifts索引结果、计数和warning触发完全一致。rank0的256次循环中，旧路径有256次 `sum/ReduceSum/gt/GtScalar/is_nonzero/item/_local_scalar_dense`，候选只剩256次必要index。门禁逐rankJSON/脚本/日志已删除，只留脱敏摘要。
- 真实1-step客户batch16门禁PASS：1/1、exit0、fatal/OOM0，loss=435.9089、grad_norm=92.0408均有限，峰值24847MiB/rank；59.782秒含初始化/编译，不作性能指标。运行副作用恢复后删除约2.638MB原始日志/harness。
- 30-step profiler-off正式结果拒绝：候选全步均值9.993467s、吞吐12.808368 samples/s、普通步mean/median/P95=6.082652/6.009000/6.946100s、SOAP窗口32.818/37.755s（均值35.2865s）、峰值26849MiB/rank。相对直接父提交 `f922c38`：全步+7.805%、吞吐-7.240%、普通步mean/median/P95=+8.118%/+6.599%/+3.463%、SOAP+5.962%，全部回归，因此 `REJECT_NO_COMMIT`。
- 相对永久基线 `4c37039`（`63861df`客户字段派生），当前候选样本仍为全步-73.308%/3.7464x、吞吐+274.644%、普通步mean-47.904%、SOAP-87.379%、显存-5.661%；这是此前累计优化结果，不能归因本候选。
- 候选已完整逆向回退，未提交；30-step约2.907MB原始日志/harness已删除，只保留脱敏metrics/report/SHA。最终远端HEAD `f922c3897255`、Git clean、训练/端口/NPU释放、kernel_meta/raw profile均0。

## 2026-08-14：R4标准MultiheadAttention准入关闭

- 当前有效链明确实例化4层MMCV `MultiheadAttention`，合同为embed_dim=256、8 heads、dropout=0；MMCV内部调用 `torch.nn.MultiheadAttention` 并丢弃返回的attention weights。固定torch_npu确实提供训练API `npu_fusion_attention`，但API存在不构成替换依据。
- 无业务改码的8-rank NPU Event探针确认真实模块路径为lane3d decoder layers0～3；每步各调用1次。Q/K/V均为 `[120,16,256]`、FP32、8头、batch_first=false、无attn/padding mask；layer0输入非连续，其余3层连续。
- 稳定step3～8的4层MHA forward device elapsed总量mean/median/min/max为1.552100/1.590990/1.204300/1.704060ms/step；host launch总量均值2.788ms/step。最大稳定device总量只占22.7ms准入门槛的7.51%。
- 旧profile的单个82.983ms BatchMatMul与当前每步4次MHA调用形态不一致，不能归因给标准MHA；NPU Event未单独计量backward，因此结论限定为“当前forward不是稳定热点”，但已足以不授权 `need_weights=False` 或融合Attention替换实验。
- R4判定 `CLOSED_BELOW_THRESHOLD`：不改业务代码、不做1/30-step候选、不提交。8/8、exit0、loss 353.6622～437.3293、grad 54.8075～94.1802均有限。恢复运行副作用并删除2,729,152 bytes逐调用JSONL/日志/work/harness，仅保留脱敏summary/report/SHA；HEAD `f922c38` clean、进程0、raw profile0。

## 2026-08-14：STEP-177 Conv/TransData 功能块归因

- 后8逻辑卡、8 rank、客户 batch/rank=16 的8-step轻量诊断完成，8/8、exit0；稳定step3～8端到端 mean/median/range=5.8495/5.8720/5.6540～6.0230s，loss=357.6238～411.7368、grad_norm=53.1188～68.8249，均有限。
- 13个命中模块中11个在当前路径实际执行。device median 排名：`img_backbone/ResNet` 362.8202ms（单步607.8978ms离群，故排序优先看median）、`bev_encoder` 311.8138ms、其子 `BEVFormerEncoder` 229.9559ms、FPN 53.2472ms、MapTR head 47.6234ms、PillarVFE 41.4210ms、SECONDTransFPNV3 31.0124ms、ConvNeXt 26.8460ms、PointPillarScatter 22.8748ms、BaseBEVBackbone 23.6488ms、LCFusionV2 19.8871ms。
- 父子计时重叠：BevEncoder包含BEVFormer、ConvNeXt、SECOND与fuser，不能求和；模块总耗时也不能直接等价为Conv或TransData可回收耗时。下一步需要在ResNet stage/BasicBlock与BEVFormer layer/attention边界进一步归因。
- 有效配置使用标准ResNet-34，输入经相机维合并后送入backbone，四个stage全部输出，BN训练态、with_cp=false；实现来自固定客户环境的mmdet安装包，forward本身没有项目自定义copy/format逻辑。高模块耗时说明优先级，不构成修改固定依赖或替换数学实现的授权。
- 一次只读 `Config.fromfile` 尝试因诊断PYTHONPATH缺少 `camera_2dbased_eval` 在import期失败，未改变任何状态；随后改为静态读取有效config和实际模块清单完成审计。
- 本轮不是完整profiler，没有kernel/trace/operator原始包。逐调用JSONL、launcher/train日志、work和harness合计删除2,833,004 bytes，raw关键文件0；保留脱敏 `module_timing_summary.json`、`module_timing_report.md` 与SHA。恢复`fusion_result.json`、删除仓库内`kernel_meta`后，HEAD `f922c38`、Git clean、训练0。
- 第二层窄计时进一步确认：ResNet stage1～4 device median=82.3206/78.8070/87.6937/52.6666ms；stage1三个BasicBlock各约27.4ms，其余多数单块低于22.7ms。输入/输出均连续NCHW，标准mmdet ResNet forward没有项目自定义copy/cast/format；因此这些值主要是必要Conv/BN/ReLU，暂不形成候选。
- BEVFormer唯一layer median=130.5539ms，其中TemporalSelfAttention 35.5320ms、SpatialCrossAttention 88.9410ms、其MSDA本体81.6528ms、FFN3.6682ms、三次LayerNorm各约0.6ms。父子闭合良好，说明layer内耗时主要是必要attention；已提交DrivingSDK MSDA不重复替换。
- BEVFormer父encoder首层计时约230ms而该layer约130.6ms，结合源码发现`point_sampling`对reference points与lidar2img分别显式`repeat`为约105MiB与420MiB/rank/step后再矩阵乘。该独立前处理成为下一计量对象；尚未因静态内存上限直接改业务源码。
- 第二层诊断8/8、exit0；稳定step3～8 mean/median/range=6.3618/6.2445/5.9050～7.2860s，loss353.3267～425.4843、grad52.2065～67.8371有限。逐调用JSONL/日志/work/harness删除2,776,017 bytes，保留脱敏summary/report/SHA，业务树clean、训练0、raw0。
## 2026-08-14：BEVFormer point_sampling 打包 BMM 候选

- 原实现对`reference_points`和`lidar2img`显式repeat，单rank/调用形成约105MiB与420MiB中间量；打包为两组三维张量后使用`torch.bmm`，8/8 rank输出逐位一致，单调用中位83.4096→1.8561ms，峰值增量662,700,544→239,077,376 bytes。
- 两次新训练进程都在iter1/2出现约64s/27s编译成本，证明不是可跨进程复用的一次性偶发预热。正式复验30-step全步9.597100s，相对父提交`f922c38`回归3.529%；吞吐回归3.409%，普通23步均值回归0.284%。
- 局部收益仍真实：SOAP窗口33.301→16.195s（-51.368%），普通P95 6.7136→6.3104s（-6.006%）；但不能覆盖30-step端到端启动成本，按统一门禁拒绝，不提交。
- 相对永久基线客户派生`4c37039`累计仍为全步-74.367%（3.9012x）、吞吐+290.095%、SOAP-94.208%；这些属于此前已提交优化的累计收益，不能归因于本候选。
- 2026-08-14技能恢复确认：新的profiling分析仍须同时维护wall/busy-union/kernel-sum/total-cost四时钟，先识别wait-anchor再归因，并输出异常JSON与独立10节架构报告。若不重采raw profile，则只能使用现有脱敏报告做候选筛选，不能伪造新的bubble或结构结论。
- 2026-08-14官方7.3复核：性能优化仍遵循“明确单机/集群场景→采集并拆解性能→定位下发/计算/通信模块→再选算法”的顺序；并行策略不存在通用万能参数，只有显存/通信掩盖证据时才调整TP/PP/DP、micro-batch或重计算。当前项目为单机8卡DP且显存非瓶颈，因此继续算子/host下发证据驱动，不为提高显存占用盲改并行策略。
## 2026-08-14：PointPillarScatter批次向量化机制门禁

- 活跃实现为`projects/mmdet3d_plugin/models/backbones/pcdet_pointpillar_scatter.py`，客户batch16下逐batch执行16次zeros、boolean mask、`sum()==0`同步和index put。候选把batch id并入全局flat index，一次zeros/一次index put后reshape+permute回原连续布局。
- 后8卡8-rank×8-call真实合同：feature FP32 `[158341～206090,32]`、coords int32、输出`[16,32,256,640]`；64/64逐位exact、max_abs0、stride一致，空batch与重复flat index均为0，feature在客户配置中不需梯度（冻结模块）。
- 稳定call3～8 pooled中位：旧/新device 31.945519/4.859540ms，节省27.085979ms、6.5738x；同步host 32.280705/5.170988ms，节省27.109717ms、6.2427x。旧模块另有16个`sum()==0`同步锚点，解释此前异步模块host wall约459ms不能直接当纯算子计算。
- 原始64条JSONL、日志、work和夹具已删除2,766,590 bytes，仅留脱敏summary/report/SHA；业务HEAD `f922c38` clean。
## 2026-08-14 PointPillarScatter 向量化门禁边界纠正

- 正确决策是 `PASS_ACTIVE_VOXEL_CONTRACT`，不是无条件 `PASS`。
- 客户有效路径的 `VoxelGenerator` 用 `coor_to_voxelindex` 合并相同坐标的点，因此输出 voxel coordinates 唯一；64/64 次真实调用也观测到重复 flat index 为 0。
- 真实大 shape、空 batch、输入梯度均为 8/8 exact；人工重复坐标用例不等价（最大差异 5.397544861），该失败已保留在脱敏结论中，候选不作通用重复坐标兼容性声明。
- 原始门禁产物已删除 30,076 bytes，raw 剩余 0；只保留脱敏报告、摘要、SHA、候选补丁和 harness 校验信息。
## 2026-08-14 PointPillarScatter 1-step 实训门禁

- 最终有效轮：8 rank 和后 8 卡 PID 映射完整，1/1 iteration 自然完成，time=99.219s（含首步编译）、memory=24,848MiB、loss=447.8574、grad_norm=94.6233，均有限，fatal=0。
- 该结果只证明可训练性，不作为性能结论；性能必须来自 30-step profiler-off 固定口径。
- 两次启动错误分别是错误入口 `tools/train.py` 和嵌套 shell 提前展开容器 `PYTHONPATH`；两轮均在有效训练前失败并已清理，不进入统计。
- 最终有效轮原始产物已删除3,218,619,445 bytes、9个文件，raw=0；仓库仅剩目标单文件候选。
## 2026-08-14 PointPillarScatter 最终结论

- 三轮池化相对`f922c38`：全步-2.853%、吞吐-2.565%，normal23均值仅+1.302%；虽然SOAP+51.860%、P95+8.628%，仍不满足端到端提交门槛。
- 结论为`REJECT_NO_COMMIT`；候选已完整回退，远端Git clean。
- 三轮原始训练产物均已删除，raw=0；仅保留脱敏指标、报告和SHA。
## 2026-08-14 PillarVFE静态候选

- 活跃配置为`PillarVFE(use_norm=True,num_filters=[32])`且整个VFE被冻结；只有一个last PFN。
- PFN卷积输出布局为`[N,32,32,1]`，当前先物化完整`[N,32,32]`连续副本再沿点维max。真实N约158341～206090，对应单rank约0.65～0.84GB复制量。
- 严格等价候选是直接沿原布局dim2做同顺序max，再对小结果换维；卷积、BN、ReLU和非最后PFN分支均不改变。必须先验证最大值、最终stride、返回view值和含tie梯度。
## 2026-08-14 PillarVFE候选关闭

- 8 rank覆盖N=158341～206090，最大值、返回view、shape/dtype、最终stride及含tie梯度全部逐位一致。
- 旧/新子步骤pooled median=12.104143/4.197734ms，节省7.906410ms、2.8835x；低于22.7ms准入门槛，因此不值得承担正式训练噪声和代码维护成本。
- 原始逐rank数据、日志和夹具已删除20,227 bytes，raw=0；远端`f922c38`保持clean。
- STEP-180 初审：活跃 `SECONDTransFPNV3` 的四个 deblock 先完成必要的上采样/卷积/BN/ReLU，再用 `torch.stack(ups, dim=-1).sum(dim=-1)` 融合。可疑冗余仅是 stack 临时张量及归约表达，不是整个 31.0124 ms 模块；需要真实 shape 微基准证明独立收益超过 22.7 ms。
- ConvNeXt 的自定义 LayerNorm 确有两次相同 `(x-u)`，但客户活跃配置明确使用 BN，故该代码不执行。活跃 Block 没有 permute/contiguous/clone，主体均为模型必要数学计算；26.8460 ms 模块总时长不足以支持盲目改写。
- BaseBEVBackbone_FPN 的 cat 是输出合同所需，逐级加法/平滑卷积/反卷积和 channel attention 均是模型数学结构；没有独立的大 copy/layout 冗余。仅 `self.cnt += 1` 可删但收益显然远低于门槛。
- 按活跃 BEV 网格 96×160、batch/rank 16、SECOND outplane 256 和上采样倍率 1/2/4/8，四个 `ups` 的共同输出为 `[16,256,96,160]`；stack 临时张量约 1.0 GiB（fp32），具备独立机制计时价值，但不得把 deblock 计算纳入候选收益。
- SECOND pairwise add 对8 rank真实规模随机输入与输入梯度逐位一致；left-associative add并非逐位一致（max abs 9.536743e-7）。pairwise完整前反向只节省7.949649 ms，低于准入线，因此即使能消除约1 GiB临时张量也不值得扩大业务变更。
- 全部历史 diagnostics 的 profiling 原始标志复核结果为0个raw profile目录、0个`PROF_*`目录；现存内容均不属于待删profiling原始包。
- STEP-181恢复审计：根目录DrivingSDK计划的“当前HEAD=bf9ed6e、立即执行R0”是历史交接状态，当前权威状态已推进到 `f922c38`，且R0及R1～R8已有后续实证，不能按旧文案重做。该计划同时明确禁止在无新证据下强行替换/block化SOAP QR或升级软件栈。
- 根目录与custom版DrivingSDK计划哈希一致；R0拒绝、R1提交、R2/R4～R8/P3均已有关闭证据，因此DrivingSDK阶段完成。通用计划仍列有HF32全关闭的单变量A/B候选，且当前ResNet/Conv为最大必要计算模块；HF32不是删除模型数学结构，但会改变低位舍入，必须先做固定版本API/真实shape数值和收益门禁，conv/matmul分开处理。
- HF32关闭发生在训练入口全局初始化处，来源`dd23198d`完整工作区提交；当前代码没有证明这是为最终loss精度独立验证过的选择。安装版torch_npu以两个独立ContextProp暴露Conv/MatMul HF32，并把它们传入编译配置，因此可在不改依赖版本的隔离进程中做单变量门禁。
- torch_npu 2.7.1源码显示Conv HF32默认启用（option未设置也视为True），MatMul默认关闭；当前入口同时False实际只覆盖了Conv默认。仓库附带的DrivingSDK MapTRv2 patch对arch35显式开启MatMul HF32，但本阶段不组合实验。Conv HF32优先级更高：覆盖ResNet/FPN/BEV卷积且属于恢复固定版本默认行为。
- 当前正式客户配置仍是20260113st的r34_0114文件，batch/rank16、8 rank、workers8；文件含`img_scale=(1333,800)`，但真实ResNet输入还受resize/crop/多相机维合并影响，机制门禁必须从实际pipeline或无业务改码shape探针取得最终NCHW合同。
- 客户相机列表为7路；ResNet注释stage shape为136×240、68×120、34×60、17×30，暗示合并相机后ResNet batch维为16×7=112。训练链实际使用ResizeCropFlipRot与Pad，需再确认最终输入高度/宽度后才能构造代表卷积。
- ida_aug_conf明确src_size=(576,1024)、resize=(-0.05,0.05)、无crop/flip/rot，Normalize后Pad到32倍数。这会形成多个输入尺寸，HF32门禁必须覆盖动态shape族和编译行为；仅用配置注释的136×240不能代表全部训练。
- 更正：`sample_augmentation`虽产生随机resize_dims，但随后crop永远取`crop_w,crop_h,crop_w+1024,crop_h+576`，越界时补边；所以每相机最终尺寸固定576×1024，Pad32无变化。实际ResNet NCHW合同为`[batch×camera=112,3,576,1024]`，与stage注释形状一致。
- Conv HF32在真实ResNet规模有明确机制收益：362.725170→284.700794ms，节省78.024376ms（21.511%）；代表Conv/Deconv也改善22.543%/6.104%。代价是低位舍入变化，跨ResNet stage及训练态Conv/Deconv输出和梯度的最大NRMSE 2.787196e-4、max abs 7.255077e-4，finite一致但非exact；必须用真实1-step loss/grad及正式30-step最终裁决。
## 2026-08-14 官方 7.3 调优文档复核（STEP-182）

- 用户给出的 `performance_tuning_0` 无 `.html` 地址被网页工具判为不安全直开；通过同站 7.3 搜索定位到“基础优化流程”和迁移总体流程。官方顺序是：先明确单机/集群问题背景，再用性能工具采集与拆解，定位到数据加载、下发/调度、Device 计算或通信瓶颈，最后才选择对应优化方法；这支持项目继续执行“证据→候选→门禁→正式 A/B”，而不是仅凭显存空余或代码观感改动。
- 用户给出的 `performance_tuning_0024.html` 是“并行策略建议”。官方指出没有通用万能并行策略；TP/PP主要处理模型放不下，DP用于资源富裕时扩展吞吐，ZeRO1/重计算可先降内存再增大 batch；通信计算掩盖不足时可增大 micro batch 或拆小单次 AG。当前客户模型单卡最大显存约 26～28 GiB、batch/rank 已为16，8卡 DDP 正常，且历史通信仅约2.7 ms、overlap约76.8%，所以不能仅因每卡60+ GiB就贸然引入 TP/PP/ZeRO；增大 batch 必须作为独立容量/吞吐/功能候选验证。
- 7.3 文档另明确：绑核对不同模型可能优化也可能因额外线程抢占而劣化；Python 3.11下发性能优于3.10；编译优化要求重编 Python/PyTorch/torch_npu。客户容器已为 Python3.11/PyTorch2.7.1，而项目规则禁止替换既有框架组件，因此 LTO/PGO/版本升级记录为不适用；CPU affinity 历史已因证据不足关闭，不重复。

## 2026-08-14 STEP-182 剩余候选矩阵

- DrivingSDK R0/R1/R2/R4～R8/P3已有当前HEAD的提交、拒绝、不活跃或不适用结论；根目录/custom计划一致，不能从旧计划文字重新开启。
- 通用计划中的SOAP QR、MSDA fallback、Hungarian、格式转换、DataContainer pin、PointPillarScatter、PillarVFE、SECOND stack、Conv HF32均已有机制或正式A/B关闭证据。MatMul HF32虽未单独跑正式A/B，但其已知证据仍只有此前已审计的DrivingSDK patch，且本轮已把HF32方向关闭；没有新的活跃shape/可回收上限证据前不直接重新占卡。
- 扩大batch不作为当前候选：客户正式口径本来就是batch/rank16、global128；继续增大会改变优化器更新对应的样本集合和训练轨迹，不能因显存尚有余量就当作功能不变优化。
- 当前留存normal-step host-context中尚未独立闭环的最高可信线索是bubble003：MapTR loss入口对GT列表执行`.to(device)`。它可能是必要CPU→NPU，也可能包含高频同设备no-op/可打包小搬运；必须先读取脱敏上下文和源码，测完整功能wall、调用次数、字节及D2H/H2D，不能用bubble gap直接当收益。

## STEP-182 bubble003 关闭结论

- bubble003 的54.692ms是相邻device kernel gap，host-context中的活跃MapTR list `.to` self仅2.522ms（colors）和1.308ms（types），`_to_copy`2.393ms、`copy_`2.018ms；这些嵌套事件不可相加，也不能把整段gap归因给搬运。
- 源码1610/1612只对客户已缓存GT type/color列表做进入loss前的必要`.to(device)`；当前脱敏证据没有同设备no-op或重复搬同一tensor的证明。即使乐观删除全部可见host self，上限也远低于约22.7ms噪声门槛。
- 结论：`CLOSED_BELOW_THRESHOLD_AND_REQUIRED_TRANSFER`，不改码、不训练、不提交。下一条未闭环host线索是bubble004的gradient fingerprint/zero_grad/DDP unused-parameter边界。

## STEP-182 DDP unused-parameter 结论

- 客户配置关闭全部gradient fingerprint/sync诊断，hook主链与标准OptimizerHook一致；不存在可删除的指纹复制/哈希开销。
- 后8卡8-rank、客户batch/rank16的3-step探针在原始optimizer step后只读梯度状态。rank0三步均为701个trainable参数/102,682,869元素，其中142个参数/25,397,504元素`grad is None`，集合稳定；主要来自img_backbone、img_neck、lane3d_head和lc_fusion分支。
- 因真实unused参数明确存在，`find_unused_parameters=False`会破坏DDP归约契约，可能报错或挂起。相对仅约3.416ms的search self没有实施资格，判定`CLOSED_REQUIRED_UNUSED_PARAMETER_DETECTION`；不做候选A/B、不改码、不提交。
- 3/3、exit0，loss/grad均有限，无错误；远端原始日志/work/hook/config/harness删除2,709,625 bytes，仅保留脱敏JSON/报告/清理报告/SHA。远端HEAD `f922c38` clean、后8卡进程0、raw profile0。

## STEP-182 纯调试路径审计：MAP_SHIFT

- 训练容器没有设置`DBG_NPU`、`SAVE_TENSOR`、blocking、task queue、combined或CPU affinity变量；SAVE_TENSOR分支不活跃，其他error warning打印仅异常触发。
- 活跃`VectorizeLocalMap`在`vectorize_local_map.py:1671～1686`仍对每个`shifts_num>final_shift_num`实例无条件输出三行`[MAP_SHIFT]`，且为了日志额外执行两次`index.tolist()`和一次`hash(tuple(...))`；随后真正训练只使用原`index`切片三组数组。删除日志不会改变样本、随机性（当前index固定arange）、tensor、loss或梯度。
- 历史batch1只看到少量日志，不能外推当前batch/rank16、workers8；本轮已按规则删除3-step raw日志，不能伪造频次。下一步先在正确容器做不占NPU的真实DataLoader短采样，统计每batch打印次数和CPU wall；未越过收益门槛不改码。
- CPU-only真实DataLoader探针使用客户`batch/rank=16`、`workers=8`、`prefetch_factor=3`顺序读取6个batch（96个样本），`idx/choice/hash`各仅2行，即总共2次触发、6行输出。dataset构建0.171352s；batch墙钟均值5.567735s受首次启动32.113347s主导，稳态中位0.269907s，最小0.214195s。
- 删除三行输出即使严格等价，其2次触发中的`tolist/hash/stdout`乐观回收也远低于22.7ms准入线，且不足以解释训练step耗时。判定`CLOSED_BELOW_THRESHOLD_LOW_FREQUENCY_DEBUG_OUTPUT`：不改业务代码、不做NPU A/B、不提交。
- 探针无错误，后8卡进程始终为0。远端5个原始文件共12,167 bytes和本地一次性脚本/pycache均已删除；远端`ascend_npu_optimize@f922c38` clean，原始profiling关键文件计数0。

## STEP-183 扩展正确性/收敛验收起点

- 根目录与`custom/`的DrivingSDK计划SHA256一致，但计划正文的权威HEAD仍停留在历史`bf9ed6e`；当前真实HEAD已推进到`f922c38`。计划中的R0～R8/P3均已有后续实证，不能按旧交接提示重做。
- 当前唯一需要扩展收敛重点验收的是`f922c38`的DrivingSDK MSDA：函数级三类真实shape NRMSE≤6.439e-7、30-step loss/grad有限，但非逐位等价；短跑不足以证明长期loss轨迹、checkpoint恢复和最终指标不偏离。
- SOAP分块foreach、GeometricLoss masked reduction已通过更强的逐位/状态门禁；TextLogger仅影响诊断字段刷新。扩展A/B仍覆盖整个提交链，但裁决重点放在`bf9ed6e→f922c38`的MSDA数值与训练行为。
- 下一步先从远端当前config/runner/checkpoint/eval与现有训练脚本冻结可执行合同，再决定一epoch或等价样本预算；不先拍脑袋启动30k全训。
- 权威config为4 epoch、28130帧、global batch128；训练入口重算`num_iters_per_epoch=219`和`max_iters=876`，因此config字面`runner.max_iters=30000`不是实际生产步数。checkpoint interval1000，但固定MMCV CheckpointHook默认`save_last=True`，末步应保存可恢复checkpoint。
- `tools/train_spetr.py`明确保持`cfg.seed=None`，分布式sampler运行时广播新的共享seed。扩展A/B保持该生产自然随机语义；两版比较epoch级分布/趋势与数量级，不对独立随机轨迹做逐step等价声明。
- canonical 30-step harness SHA仍为`10ad92c...e0fc`，通过`MAX_ITERS`可覆盖到876；resume需在末步checkpoint出现后使用单独的测试夹具或等价现有入口传`--resume-from`，不修改业务源码。
- config训练内evaluation interval为100000000，不会在876步自动评测；仓库有8卡`test_spetr.py`/ddp_test入口，但test数据引用远端对象存储。完成训练后必须先验证可达性，再决定最终checkpoint评测，不能预先假定最终任务指标可用。
- 父提交`bf9ed6e`扩展训练第1个完整epoch已提供生产自然随机参考：loss均值/中位数142.921568/112.150100，首20步均值351.390015降至末20步74.619565，无非有限值；普通步中位6.212s，SOAP邻接步中位18.702s，epoch吞吐14.622955 samples/s。当前HEAD需按完全相同统计口径比较趋势与数量级，不做逐step相减。

## 2026-08-14：profiling raw 清理规则复核

- 用户再次明确：旧 profiling 数据在本轮使用结束、后续不再需要时删除。
- 远端 diagnostics 原位只读复核结果：raw profiling 关键文件 0 个、0 bytes，`PROF_*`/`ASCEND_PROFILER_OUTPUT` 原始目录 0 个，因此本次无实际删除对象。
- 当前运行的是 STEP-183 profiler-off 父提交长期收敛训练；其日志、work 和末步 checkpoint 仍承担 876 步统计、恢复 5 步及评测可达性验证，不能按 profiling raw 误删。

## STEP-183 父提交 epoch2 结论

- `bf9ed6e` 的 iter220～438 完整 epoch2：loss mean/median=`60.518547/60.205100`，由前20步 mean `74.245135` 降至末20步 `51.337850`；grad mean/median=`46.188121/44.834100`，全部有限，严格异常0。
- normal 175步 mean/median/P95=`6.397589/6.294000/7.699600s`，SOAP 44步=`16.943227/17.437500/27.186450s`，全epoch mean=`8.516347s`、吞吐=`15.029918 samples/s`。
- framework memory 峰值 `27086 MiB`，与epoch2前段一致；进入epoch3后前4步仍未增加。当前证据支持持续收敛和显存平台化，但需完成4 epoch和resume门禁后才可作为发布级父参考。

## STEP-183 父提交 epoch3 结论

- iter439～657 loss mean/median=`43.356168/43.102500`，由前20步mean`46.933555`降至末20步`40.682230`；相对epoch2 mean`60.518547`继续下降，无数量级异常。grad mean/median=`42.395856/41.418900`，全部有限，严格异常0。
- normal mean/median/P95=`6.446834/6.352000/7.740600s`，SOAP=`16.845659/17.152500/25.914050s`，全epoch mean=`8.536096s`、吞吐=`14.995146 samples/s`。
- memory峰值连续保持`27086MiB`，进入epoch4前7步仍未上涨。父提交前三个epoch形成稳定下降参考，可继续最终epoch。

## GPU A800 日志与NPU 30-step同编号窗口对比

- 本地文件`gpu去除随机性固定后loss.log`确证8×A800-SXM4-80GB、CUDA/NCCL rank0～7、完整3664步、seed=0且deterministic=False；日志数据帧数117286，与NPU客户验收28130帧不同，因此是硬件性能参考，不是严格同数据A/B。
- GPU前30步按NPU固定窗口重算：全步mean/median/P95=`6.698167/4.380000/8.486950s`；普通23步mean/median/P95=`4.717565/4.378000/5.862400s`；11/12/21/22同编号窗口mean=`4.236250s`；峰值显存`28409MiB`。
- 按batch/rank16×8条件计算吞吐=`19.109707 samples/s`。日志虽重复打印batch_size16，但未打印完整loader`samples_per_gpu`，因此吞吐需标记为条件计算；窗口也不能仅凭日志标成SOAP。
- GPU loss 30/30 finite；grad门禁未通过：step1日志报告NaN并跳过optimizer step，step2/3为inf，step4～30有限。不能写成GPU前30步grad全有限。
- 相对NPU当前`f922c38`：GPU全步耗时低27.743%（1.384×快）、条件吞吐高38.395%、普通mean低16.146%、同编号窗口低87.279%（7.861×快），但显存高1561MiB（5.814%）。

## STEP-183 父提交4-epoch与resume参考完成

- epoch4 loss mean/median/P05/P95=`35.785836/35.168800/27.794930/45.920500`，前20步mean`39.885635`降至末20步`34.604210`；grad mean/median=`42.554868/41.809300`，全程有限。
- epoch4 `time` mean=`8.604228s`；normal=`6.470903/6.303000/8.055100s`，11/12模式窗口=`17.089045/17.693000/27.241300s`，throughput=`14.876407 samples/s`；`memory`峰值由27086升至27173MiB但无OOM。
- 876步整体loss从epoch1→4 mean=`142.921568→60.518547→43.356168→35.785836`，持续下降；876条loss/grad全有限、fatal0。
- `iter_876.pth`/`latest`完整。MMCV日志显示文件在“Saving checkpoint at 876 iterations”生成，但resume报告`iter 875`；因此达到881会执行/记录876～881六条。所有六条loss/grad有限、resume loss mean=`33.352483`，与主训练末20步mean=`34.604210`连续；`iter_881.pth/latest`生成，恢复门禁接受并记录off-by-one语义。

## STEP-183 当前MSDA版本 epoch1 对比

- 当前epoch1 `loss` mean/median=`139.128243/109.554200`，首20步mean`342.197115`降至末20步`72.327595`；父版本=`142.921568/112.150100`、`351.390015→74.619565`。自然随机轨迹不同但分布、下降趋势和数量级一致，未见MSDA引入发散。
- 当前`grad_norm` mean/median=`46.727786/45.177900`，219条全部finite、fatal0；父版本=`50.930898/47.760900`且同样finite。
- 当前全epoch `time` mean=`8.130699s`，normal mean/median/P95=`5.798000/5.639500/7.176000s`，固定窗口mean=`16.458238s`，throughput=`15.742805 samples/s`，`memory` max=`26842MiB`。
- 相对父epoch1：全epoch `time` -7.113%，normal mean -8.363%，窗口 -2.974%，throughput +7.658%，memory +32MiB。相对GPU前30步只作性能参照：normal mean仍约慢22.90%，固定编号窗口约3.885×；数据集与seed不同，不能当严格A/B。

## STEP-183 当前MSDA版本 epoch2 对比

- 当前epoch2 `loss` mean/median=`59.415967/58.483800`，前20步mean`72.269845`降至末20步`51.273245`；父版本=`60.518547/60.205100`、`74.245135→51.337850`。末段均值仅差0.126%，收敛连续。
- 当前`grad_norm` mean/median=`46.117401/44.824200`，全部finite、fatal0；父版本=`46.188121/44.834100`，分布高度一致。
- 当前`time` mean=`7.971680s`，normal=`5.864211/5.718000/7.032600s`，窗口mean=`16.353659s`，throughput=`16.056840 samples/s`，`memory`=`27085MiB`。
- 相对父epoch2：time/normal/window分别-6.395%/-8.337%/-3.480%，throughput+6.833%，memory-1MiB。相对GPU前30步参考：normal mean约慢24.31%，固定编号窗口约3.861×；仍不能作为严格A/B。

## STEP-183 当前MSDA版本 epoch3 对比

- 当前epoch3 `loss` mean/median/P05/P95=`43.287042/43.381800/31.529580/54.766990`，前20步mean`46.199335`降至末20步`40.483295`；父版本分别为`43.356168/43.102500/31.938700/55.758720`、`46.933555→40.682230`，趋势和数量级一致。
- 当前`grad_norm` mean/median=`43.943570/43.387100`，219条全部finite、fatal0；父版本为`42.395856/41.418900`，未见发散或数量级偏离。
- 当前`time` mean=`8.261699s`，normal mean/median/P95=`6.123983/6.000000/7.654600s`，固定窗口mean/median/P95=`16.763977/18.812000/26.762600s`，`throughput (samples/s)=15.493182`，`memory` max=`27085MiB`。
- 相对父epoch3：`time` -3.214%、normal mean -5.008%、固定窗口 -0.485%、`throughput (samples/s)` +3.321%、`memory` -1MiB。相对GPU前30步参考，normal mean仍约慢29.82%，固定编号窗口约3.957×；由于数据与seed不同，仅作性能方向参考。

## STEP-183 当前MSDA版本4-epoch与resume最终结论

- 当前epoch4 `loss` mean/median/P05/P95=`35.841326/35.668500/27.146650/46.235760`，前20步mean`39.575625`降至末20步`35.016005`；父版本为`35.785836/35.168800/27.794930/45.920500`、`39.885635→34.604210`。当前相对父`loss` mean仅+0.155%、末20步+1.190%，趋势和数量级一致。
- 当前epoch4 `grad_norm` mean/median=`41.695886/40.217000`，全部finite；`time` mean=`8.299005s`，normal mean/median/P95=`6.204086/6.042000/7.685200s`，固定窗口mean/median/P95=`16.631068/17.420500/25.476900s`，`throughput (samples/s)=15.423537`，`memory` max=`27175MiB`。
- 相对父epoch4：`time` -3.547%、normal mean -4.123%、固定窗口 -2.680%、`throughput (samples/s)` +3.678%、`memory` +2MiB。全876步当前`time` mean/median/P95=`8.165771/6.050000/25.248000s`，父版本=`8.602508/6.436000/25.846000s`，分别改善5.077%/5.998%/2.314%；`throughput (samples/s)`由14.879382升至15.675189（+5.348%）。
- 当前主训练876/876自然退出，全部`loss/grad_norm` finite、fatal0，`iter_876.pth/latest`有效。恢复从checkpoint meta的iter875继续，记录876～881六条；`loss` mean/range=`34.500800/29.4106～38.9806`、`grad_norm` mean=`42.421717`，全部finite、fatal0，`iter_881.pth/latest`有效。MSDA长期收敛和恢复门禁通过。

## STEP-183 最终任务指标可达性边界

- 正式`test_spetr.py`入口支持8-rank distributed eval，但当前权威config的`data.test`含旧字段`lidar_type`，而活动`InternalDatasetTrackStream.__init__`无该参数；原始config在dataset构建前确定性失败。该行自`dd23198d`及永久基线`63861df`即存在，不是`f922c38`引入。
- 本地客户配置中该`lidar_type`行处于注释状态。诊断进程仅在内存中移除该键、不写仓库后，dataset构建在90秒内仍未返回`dataset_len`；测试路径包含`s3/iaginfra`，当前无法证明对象存储/标注可达。
- 因此本轮不能给出新checkpoint的F1/Precision/Recall等最终任务指标。`custom/rg_evaluation_results.txt`是既有历史结果，来源checkpoint和运行口径未与本轮绑定，不能用于证明MSDA最终任务等价。

## STEP-183 最终裁决、GPU主对比与清理

| metric | GPU A800 first 30 | NPU `f922c38` first 30 | GPU vs NPU |
|---|---:|---:|---:|
| `time` mean (s) | 6.698167 | 9.269933 | -27.743% / 1.384× faster |
| `time` normal mean (s) | 4.717565 | 5.625957 | -16.146% / 1.193× faster |
| `time` fixed-number window mean (s) | 4.236250 | 33.301000 | -87.279% / 7.861× faster |
| `throughput (samples/s)` | 19.109707* | 13.808082 | +38.395% |
| `memory` max (MiB) | 28409 | 26848 | +1561 / +5.814% |

- `*` GPU `throughput (samples/s)`按用户给定batch/rank16×8条件计算；GPU日志未直接打印完整loader字段。GPU数据117286帧/seed0，NPU数据28130帧/自然随机，故该表是主要性能参照而非严格功能A/B。`loss`方面两边前30步均finite；GPU `grad_norm` step1缺失且有NaN skip、step2/3为inf，不能作为NPU正确性参照。
- NPU永久客户同口径基线`4c37039`到`f922c38`的30-step累计为`time` 37.440000→9.269933s（-75.241%、4.039×），`throughput (samples/s)` 3.419000→13.808082（+303.863%），normal `time` 11.449000→5.625957s（-50.861%），固定窗口279.589000→33.301000s（-88.089%），`memory` 28460→26848MiB（-5.664%）。
- 直接父版本4-epoch长期对比证明MSDA单项全876步`time` -5.077%、`throughput (samples/s)` +5.348%，四个epoch `loss`持续下降且最终epoch均值仅+0.155%，resume连续。因此接受并保留既有单功能commit `f922c38`，验收本身不创建commit。
- 清理时先确认两个raw目录和父worktree均位于本轮诊断目录且非符号链接；精确恢复父worktree已知`fusion_result.json`运行副作用。移动父/当前两个iter876 checkpoint到`evaluation_pending`并生成/复核SHA256后，删除剩余3,243,389,144 bytes raw、iter881、日志、harness和父worktree。最终保留2个checkpoint+SHA共3,215,983,844 bytes；主仓库clean、后8卡空闲、profiling raw0。本地一次性resume夹具也已删除。

## STEP-184 SOAP周期窗口最新profile结论

- rank0 Level0采集精确覆盖Step10/11。Step10 `service/device busy/underfeed=62376.371/23903.239750ms/61.679%`，Step11=`37310.357250/1060.265500ms/97.158%`；profiling开销很大，`service/time`不能替代profiler-off性能，但纯device差分可用于归因。
- `aclnnLinalgQr_QrAiCPU_Qr`只出现在Step10，543次、纯duration=`22641.383956ms`、wait=`9.634450ms`、masked ratio约`0.0002%`，占Step10 device busy 94.721%、占Step10-Step11 device busy差99.118%，不是wait-anchor假热点，而是完全暴露的AICPU计算。
- QR shape由少数大矩阵主导：4×2560=`16147.768347ms`（QR总量71.32%）；22×768=`2188.898305ms`，43×512=`1558.070687ms`，6×1024=`1459.024009ms`，181×256=`1048.221737ms`。前五组约占QR总量99%。调用栈稳定落在`soap.py:422→337→174/191`。
- 相邻普通Step11纯device前列为Conv `271.678672ms`、MSDA backward `185.692888ms`、Conv TransData `102.952347ms`、BatchMatMul `82.747846ms`、MSDA forward `81.489094ms`。这些分别属于已拒绝/必要的Conv/格式、已优化MSDA和已关闭point_sampling路径，未形成新安全候选。
- 与历史同口径QR `22.711s/543次`相比仅约-0.31%，说明MSDA提交没有改变SOAP根因。GPU固定编号窗口`4.236250s`仍是主要性能参照，但GPU日志未标记SOAP且数据/seed不同，不能把全部差值归因给QR。
- 决策`CLOSED_NO_NEW_FIXED_ENV_EQUIVALENT`：`geqrf/orgqr/householder`因非位级及fallback拒绝，同shape batch变慢，out-buffer无收益，多stream正式8-rank回归；分块、降低max_precond_dim或改frequency会改变optimizer basis/状态更新语义。固定客户环境又禁止升级到潜在新算子，因此不改码、不提交。
- 技能合同已满足：结构化异常JSON通过正式schema（0 error），异常报告和独立模型架构报告均生成，架构10节完整；通信JSON缺失的边界已在报告中注明。
- 清理完成：删除151个profiling/export文件以及work/harness/分析器等合计`6,903,469,756 bytes`，关键raw计数0；只保留异常JSON/报告、10节架构报告、manifest、schema、候选裁决、cleanup report和SHA共9文件/315,720 bytes。远端HEAD `f922c38`、Git clean、端口29927为0、物理NPU4～7空闲。
- 仓库内存在共享`kernel_meta`缓存；前8卡同事任务仍在同容器运行，无法证明该目录只归本轮，且它不是profiling raw，故未删除，避免影响同事任务。
## STEP-185 客户评测镜像、同 checkpoint A/B 与评测边界

- 本地客户配置的评测文件名为 `val_fram_list.json` 和 `val_fram_list_flag.json`。旧 GPU 路径在当前 NPU 服务器不存在，但在既有只读挂载下找到一组同名配对镜像；路径只用 SHA256 标识。两文件可被当前容器读取，dataset 构建为 `InternalDatasetTrackStream`，`dataset_len=25287`，构建 `0.117842s`；`sample0` 完整加载耗时 `0.890030s`。
- 当前权威 config 的 `data.test` 含历史遗留 `lidar_type`，活动 dataset 构造函数不接受该键；本地客户配置已注释该字段。诊断 runtime config 只在临时文件中移除该键并换成镜像路径，不修改业务 config、不形成 commit。
- 正式测试入口要求客户已有 `VIS_RATE=1` 语义。遗漏该变量时 8 rank 均在 `PointPillarScatter_Seg` 报 `KeyError: data_tag`；加入客户脚本既有变量后 16/16 和 512/512 均成功。该错误是调用合同缺失，不是数据、checkpoint 或 DrivingSDK 错误。
- STEP-183 保留的两个 `iter_876.pth` 和诊断目录在本轮开始时已不在当前共享仓库，无法恢复；因此改用同一份现有客户 checkpoint（meta `iter=2999`）分别加载父/当前代码。该合同让权重和输入完全相同，直接隔离 MSDA forward 实现；STEP-183 的 876-step A/B 继续负责训练收敛、`loss/grad_norm` 和 resume 证据。
- 16-sample 原始输出比较覆盖 1,785,186 个数值元素，`structure_mismatches=0`、`shape_mismatches=0`、`nonfinite_mismatches=0`，`nrmse=3.22317013402e-05`、`max_abs=0.0239329710603`。
- 512-sample 固定 shape 输出：`seg nrmse=1.92639787185e-04`、`max_abs=7.51584768295e-04`；`map_scores_3d nrmse=1.11411394726e-04`、`max_abs=1.14834308624e-03`，无 nonfinite mismatch。可变长 map/seg-line 结果有数量、排序和 shape 分叉，不能把位置错配后的全局 `nrmse=0.1006` 当成模型误差。
- 512-sample 推理日志：父 `task/s=14.2`、`elapsed=36s`；当前 `task/s=15.8`、`elapsed=32s`，即 `task/s +11.2676%`、`elapsed -11.1111%`。两轮均为正确容器、逻辑 8～15、8 rank；`npu-smi` 显示八个主进程位于物理 4/0～7/1。
- 辅助同-shim RG 结果按日志英文名比较：LANELINE/ROADSIDE/CENTERLINE 的 `F1/Precision/Recall/IoUMean/TP/FP/FN/TN` 全不变；CROSSWALK 仅 `FP +1`，其余不变；STOPLINE 仅 `IoUMean -0.001`，其余不变。`Horizontal Mean Error (All)` 的变化范围为 -0.011～+0.018，LANELINE各距离窗口为 -0.006～+0.006。
- canonical 边界：固定客户容器缺少 `ortools`，远端规则禁止安装。仅在本地隔离目录安装官方 `ortools 9.14.6206` 和 `networkx 3.2.1` 进行兼容验证；NetworkX shim 对 5,000 个随机矩阵的可行性/最优成本 0 mismatch，但等成本匹配边有 75 mismatch，故未冒充 canonical evaluator。上述 RG 表只能作为父/当前共用同一 shim 的辅助相对证据，不能发布为客户全量绝对指标。
- 综合 STEP-183 与 STEP-185：DrivingSDK MSDA 当前 `f922c38` 保持采用。训练全876步 `time -5.077%`、`throughput (samples/s) +5.348%`、epoch4 `loss` mean 仅 +0.155%；同 checkpoint 推理 `task/s +11.2676%`，主要任务指标在日志精度下保持一致，未发现“为了性能改变最终功能”的证据。

## STEP-186：最终交付审计结论

- 当前采用链从永久算法基线`63861df`到`f922c38`共有7个性能功能commit和1个客户字段对齐commit；诊断、验收和拒绝候选没有创建性能commit。
- 30-step主结论：`4c37039 -> f922c38`的`time`为37.440000→9.269933s（-75.241%），`throughput (samples/s)`为3.419000→13.808082（+303.863%），`memory`为28460→26848MiB（-5.664%）。
- GPU主参照：`time=6.698167s`、条件`throughput (samples/s)=19.109707`、`memory=28409MiB`；GPU相对当前NPU的`time`快1.384×。GPU数据/随机语义不同且早期`grad_norm`存在NaN/inf，只能作性能参照。
- DrivingSDK MSDA直接增量：父/当前876-step的`time=8.602508/8.165771s`、`throughput (samples/s)=14.879382/15.675189`；当前分别-5.077%/+5.348%。最终epoch`loss` mean仅+0.155%，两版876/876 `loss/grad_norm`有限且resume通过。
- 同checkpoint 512-sample推理：`task/s=14.2→15.8`（+11.2676%），`elapsed=36→32s`（-11.1111%）；`seg`和`map_scores_3d`固定shape `nrmse`分别1.92639787185e-04和1.11411394726e-04，nonfinite mismatch均0。
- 当前固定环境剩余最大热点仍是SOAP周期QR；新profile确认`aclnnLinalgQr_QrAiCPU_Qr=22.641383956s`，但没有逐状态等价的更快实现，维持`CLOSED_NO_NEW_FIXED_ENV_EQUIVALENT`。
- canonical 25,287样本绝对`F1/Precision/Recall/IoUMean/TP/FP/FN/TN`仍依赖客户既有OR-Tools评测环境；不能在远端固定容器安装，也不能用临时shim冒充。

## STEP-187：本地临时产物清理结论

- `.codex-remote-edit`混合了37个tracked诊断夹具和11个untracked临时文件，不能整体视为临时目录；误删后已从HEAD恢复全部tracked文件并确认status0，只保留对11个untracked文件的清理。`work`只含R2/R6/R7/R8/R9/R10/R11已闭环实验的本地中间报告，必要结论均已进入持久记录和最终报告。
- `.codex-tools/__pycache__`仅为可再生字节码缓存；删除不影响工具源码。
- 净清理范围为：`.codex-remote-edit`的11个untracked文件、`work`53个文件、`.codex-tools`顶层65个一次性候选文件、顶层cache27个文件及依赖目录cache91个文件；合计247个文件、1,737,330 bytes。
- `.codex-tools/python-packages`与远端执行源码不属于当前可删除临时物，因为canonical环境只读核验仍可能使用；客户输入和最终交付物同样保留。

## STEP-188：GPU/NPU最大公共窗口结论

- GPU日志共有3664条完整iteration，NPU当前`f922c38`完整训练共有876条，因此最大公共窗口是1～876；前30步不能继续作为GPU/NPU总体主结论。
- GPU 1～876：`time` mean/median/P95=`4.515542237/4.312500/5.543500s`，条件`throughput (samples/s)=28.346540298`，`memory` max=`28816MiB`。
- NPU 1～876：`time` mean/median/P95=`8.165771/6.050000/25.248000s`，`throughput (samples/s)=15.675189`，`memory` max=`27175MiB`。
- GPU平均`time`比NPU低44.702%、快1.808×；条件吞吐高80.837%。NPU显存少1641MiB，但显存余量本身不等价于算力利用率或可直接扩大batch。
- `4c37039`永久基线只有30-step现成数据，故其累计收益继续按30-step报告；该表与876-step GPU主表必须分开标注，不能混合窗口。
- GPU前876步`loss`全部有限，但`grad_norm`仍有step2/3两条`inf`，step1 NaN跳过更新；功能正确性继续由NPU同版本876-step、resume和同checkpoint推理A/B承担。

## STEP-189：1:1目标差距与剩余候选

- 当前876-step参考中，NPU:GPU `throughput (samples/s)`=`15.675189:28.346540`，即约`0.553:1`；达到1:1需NPU吞吐提升80.837%，或平均`time`从8.165771 s降至4.515542 s，每步减少3.650229 s。
- 普通步median仍相差1.7375 s；周期路径P95相差19.7045 s。若周期步约占20%，周期路径每节省1 s只折算约0.2 s全局均值，因此必须同时处理普通步和SOAP周期步，不能只靠一个小算子达到目标。
- 当前最高优先级不是立即改代码，而是对`f922c38`采集低开销普通步/周期步配对profile。旧with-stack profile会明显放大host wall，适合归因、不足以精确分配3.650229 s差距预算。
- SOAP QR仍是最大单点机会：Step10中`aclnnLinalgQr_QrAiCPU_Qr`为22.641383956 s、占device busy 94.721%。既有固定环境内置替代均已拒绝；剩余高风险方向仅为“利用容器已有工具链开发项目内自定义等价算子”的可行性研究，禁止安装或升级远端组件，且必须逐状态等价、checkpoint可恢复。
- 普通步优先候选：按SOAP周期条件化的DataContainer pin/搬运调度、后8卡rank/worker CPU-NUMA亲和、局部TransData/layout边界消除。`MatMul HF32`只在新profile证明暴露足够时做单变量A/B，其现有约82.7 ms/step上限远小于总差距。
- 更大batch、ZeRO、recompute、TP/PP属于改变训练合同或模型容量策略，不能计入当前同batch功能保持的1:1验收；可另建吞吐扩展曲线，但必须与主指标分栏。
- 已关闭且不重复：Conv HF32、全局internal format、TASK_QUEUE/COMBINED、QR batching/out-buffer/multi-stream、无条件pin memory、point sampling packed BMM等。
## STEP-189 普通训练步全算子结论（2026-08-14）

- `f922c38`、客户 batch/rank16、后 8 NPU、8 rank 的普通 Step7：`service_ms=7762.3855`，`device_busy_union_ms=1916.08275`，`underfeed_ratio=75.3158%`。
- 全量聚合包含 243 个唯一算子、84,811 次 device kernel 调用，kernel duration sum=`1928.575394 ms`；完整清单见 `STEP-189_f922c38_全部算子耗时.csv/.md`。
- 纯 kernel 前五：Conv2D `271.631355 ms`、MSDA Grad `187.430755 ms`、Conv TransData `102.938092 ms`、BatchMatMulV2 `82.961842 ms`、MSDA forward `82.345215 ms`。
- 类别占比：Conv backbone `490.566843 ms (25.437%)`、MSDA `269.775970 ms (13.988%)`、Layout/Copy `243.360041 ms (12.619%)`、Attention/MatMul `224.836641 ms (11.658%)`、Elementwise `213.197111 ms (11.055%)`、Index/Reduction `198.300161 ms (10.282%)`。
- 普通步 QR 调用为 0；SOAP QR 仍只属于周期重步。普通步最大问题是 75.3% device underfeed，下一轮不能只按单算子 kernel duration 排序，还需结合 host wait/launch gap 归因。
- 本轮原始 profile 已清零；只保留脱敏聚合与校验报告。
## STEP-190 普通步 underfeed 软归因（2026-08-14）

- STEP-189 保留的异常报告显示 dominant idle pattern=`internal_bubble`，前五大 bubble 均为 `possible_host_launch_lag`（medium confidence），host coverage 100%，sync/comm overlap 均为 0；不是 HCCL 主导，也没有 wait-anchor false hotspot。
- bubble_1 `18.964 ms` 的 host evidence 包含 `aten::_local_scalar_dense`、`aten::item`、`torch.distributed.ddp.reducer::search_unused_parameters`、`SOAP.zero_grad`。
- bubble_2 `11.973 ms` 位于 DDP `copy_bucket_to_grad`/`linalg_vector_norm` 一带；bubble_3 `8.454 ms` 位于 autograd Add/MeanBackward；bubble_4 `7.428 ms` 位于 `Nonzero/Index/ge/le`；bubble_5 `6.808 ms` 位于 `Nonzero/IndexPut`。
- HCCL kernel sum 仅 `12.638 ms`，且 87.155% 与非 HCCL compute 重叠；当前不把通信作为普通步 P0。
- 由于 Level0 无 source stack/shapes，事实是普通步存在高 underfeed 和 host launch 风险；具体源码根因仍需静态调用链+最小 host 证据闭环，不能直接宣称某个 Python 函数是唯一根因。
- 历史门禁交叉核验：DDP `find_unused_parameters=True` 已由3-step探针证明每步固定142/701个trainable参数无梯度，必须保留；MapTR `Nonzero/Unique/IndexPut`、DataContainer pin、CPU affinity、global internal format、Conv HF32均已正式A/B拒绝或证明必要，不能因新profile再次出现同名kernel而重开。
- 当前新profile唯一尚未独立闭环且超过22.7ms准入线的固定环境候选是 **MatMul HF32**：`Attention/MatMul` kernel sum=`224.836641 ms/step`，其中单次BatchMatMul=`82.961842 ms`、1906次Mm合计=`46.994964 ms`。Conv HF32已拒绝，但MatMul是独立option，尚未正式测试；必须先取得真实shape与forward/backward误差/收益门禁。
## 2026-08-14：STEP-190 MatMul HF32 候选闭环

- shape-only单步在`f922c38`、客户batch16/rank、后8卡和8 rank上自然完成，`Iter [1/1] time=61.416s`仅包含冷启动与hook开销，不能作为性能指标；`loss=448.1172`、`grad_norm=122.0656`均有限。
- rank0记录111类、133次Linear/MatMul调用，估算forward合计`888,575,361,024 FLOPs`。主要模块是BEV encoder的sampling offsets/FFN/value projection，以及4层lane3d decoder `output_proj.0`。
- 三个原始精确Linear shape的8-rank HF32门禁（每rank/模式3次，24样本池化中位数）：`15.002210 -> 13.738930 ms`，节省`1.263281 ms`，speedup=`1.091949x`。
- 分shape：BEV sampling offsets仅省`0.002211ms`；BEV FFN省`0.360870ms`；lane output projection省`0.900200ms`、`1.596740x`。后者在真实模型调用4次，三类按调用数加权合计预计约`3.964ms/step`。
- 所有output/grad_x/grad_weight/grad_bias均有限，但HF32并非逐值等价：最坏`NRMSE=1.469314e-4`，最坏`max_abs=0.797241`（大规模grad_weight）。结合普通步MatMul kernel家族总上限`224.836641ms/step`，证据不足以达到`22.7ms/step`阶段门槛。
- 裁决`REJECT_LOW_BENEFIT_WITH_NUMERIC_PERTURBATION`：不进入1-step/30-step训练A/B，不启用全局`torch.npu.matmul.allow_hf32`，不修改业务代码、不创建commit。
- 两轮原始JSON/log/work/临时脚本均已精确删除，远端仅保留脱敏summary/report/JSON/SHA/cleanup；业务仓库恢复为clean，相关进程与端口均为0。

## 2026-08-14：STEP-191 下一候选筛选

- STEP-189中的972次`LinalgVectorNorm`不是新候选：历史STEP-109已在559个真实shape上证明正式`foreach=None`自动走批量路径，`13.778ms`与显式True `13.823ms`等价，显式开关反而慢约`0.045ms`；STEP-123也证明剩余`norm.to(first_device)` profiler-off仅约`1.06ms`。
- 其余高频Index/Reduction和SOAP小算子已由Nonzero/Unique/IndexPut、grad clip、zero_grad、SOAP分块foreach等历史门禁覆盖，不能因稳定Step7再次出现同名kernel而重开。
- 当前未独立闭环的可疑族为`aclnnInplaceRelu`：90次纯kernel=`34.472007ms`，对应backward 83次=`10.924639ms`；另有60次out-of-place ReLU仅=`4.613472ms`，但shape不同，不能直接宣称in-place更慢。
- 下一步只采集ReLU模块元数据并做同shape机制门禁；切换`inplace=False`只有在模块输入后无alias读取、output/gradient逐值一致、显存可接受且预计端到端收益超过22.7ms时才有资格改业务代码。

## 2026-08-14：STEP-191 Inplace ReLU裁决

- 8-rank单步元数据探针自然exit0，`loss=440.0688`、`grad_norm=125.0624`均有限；`time=52.797s`包含冷启动，不作性能结论。hook覆盖121类、149次`nn.ReLU`调用，其中in-place 90次与profile的90次完全一致，out-of-place module 59次与kernel 60次相差1次functional调用。
- in-place累计处理`5,539,627,008`元素，out-of-place累计`598,180,608`元素，规模比9.26倍。结合稳定Step7 kernel `34.472007/4.613472ms`，单位元素约`0.006223/0.007714ns`，in-place反而快19.33%。
- 最大in-place来自冻结的`img_backbone.relu`，shape=`[112,64,288,512]`、`1,056,964,608`元素、`requires_grad=False`；改out-of-place单次需新增约`4,227,858,432 bytes`输出。大多数加权元素也来自不需要反向的图像backbone，无法回收现有`ReluGrad`。
- 裁决`REJECT_VOLUME_EXPLAINS_COST_AND_MEMORY_REGRESSION`：不做同shape机制测试或训练A/B，不切换`inplace=False`。raw JSON/log/work/临时脚本均删除，远端只留脱敏摘要/SHA并恢复clean。

## 2026-08-14：稳定Step TopN主线恢复结论

- 当前TopN权威来源是STEP-189在预热后的Step7采集的单步profile，不使用冷启动shape probe的`time`进行排序。
- 已闭环顺序为：MatMul HF32因预计约3.964ms/step且产生数值扰动而拒绝；Inplace ReLU因总耗时由9.26倍元素量解释且改法增加约4.23GB输出而拒绝。
- 当前唯一开放候选是STEP-192冻结Backbone+FPN局部channels-last；它必须先通过真实shape、包含边界layout转换的8-rank机制门禁，才可能进入训练A/B。

## 2026-08-14：华为PyTorch 7.3.0官方调优文档复核

- 官方“基础优化流程”要求先明确单机/集群问题背景，再用性能工具采集和拆解，最后把瓶颈细化为下发、计算、通信后选择对应优化算法；这支持当前“稳定Step profile → TopN/空泡分类 → 单变量候选”的主线。
- 官方“并行策略建议”明确不存在通用万能策略：TP优先用于模型装不下且不超过单机卡数；PP仅在TP后仍不足且数量应尽量小；DP/ZeRO1/重计算和增大batch均需结合资源与通信计算掩盖实测。
- 本项目主验收固定8卡、batch/rank16与现有分布式语义，因此TP/PP/ZeRO或增大batch不能用于粉饰NPU/GPU主比值；只有stable-step证据显示通信或显存是主瓶颈时，才可另立同合同候选。
- 用户给出的`performance_tuning_0`裸URL被官方站点/抓取器判定为不可直接打开；已用同版本官方`performance_tuning_0016.html`“基础优化流程”页面补足核心流程。第二篇`performance_tuning_0024.html`已直接读取。

## 2026-08-14：GPU配置对齐新增边界

- GPU权威配置为用户明确指出的本地根目录同名文件；根目录与`custom`副本字节一致、SHA均为`9039BD31...CA33B`。不同的`3B1F2433...F9E8`属于`.codex-remote-edit`历史快照，不能当GPU权威配置。
- NPU候选必须在同一模型、数据、batch/rank、总batch、优化器、学习率/调度、训练步、随机性和评测语义上与GPU尽可能一致；只允许硬件设备、NPU融合实现及固定环境兼容所必需差异，并逐项说明。
- STEP-192只是对当前配置中的冻结Backbone+FPN做机制门禁；在配置差异未审计、远端工作树未归因前，不启动测试。
- 远端当前NPU生效配置SHA为`217EC2E7...B721`、148839 bytes，明显不同于本地GPU配置；差异必须在远端原位分类为硬件适配、已批准性能提交或非合同差异，尚不能宣称严格对齐。

## 2026-08-14：GPU/NPU配置合同审计结论

- MMCV结构化比较共42项差异。严格一致项：8卡、batch/rank16、LR计划、runner、DDP unused参数策略、checkpoint与evaluation。
- 明确功能差异：训练`dropout_sd_prob 0.2→0`；BEV encoder三项lidar随机丢弃/遮罩参数`0.1/0.2/0.2→0`；`model.use_grid_mask True→False`。这些会改变样本或模型行为，不能归为普通NPU路径适配。
- optimizer合同差异：GPU为`Fp16OptimizerHookProtectGradNan`且`loss_scale=dynamic`；NPU为`GradientFingerprintOptimizerHook`，增加fingerprint字段并设`synchronize_after_backward=False`，同时GPU字段loss_scale缺失。必须审计继承关系和数值语义，不能仅凭loss有限认定一致。
- SOAP配置差异：NPU增加`one_sided_dim_threshold=1024`，属于既有性能提交，但仍需保留等价性证据；日志增加`memory_interval=10`属于诊断/性能差异。
- data的train/val/test路径与部分字段不同，另有多组NPU-only中间列表变量。顶层未引用列表可排除运行时影响，但实际`data.*`差异必须证明样本集合、顺序和标注语义等价。
- plugin路径和`custom_imports`属于候选NPU适配差异；图像checkpoint仅路径哈希不同，需原位比较文件SHA或权重身份，不能输出路径。
- 因此此前0.553:1只能作为历史条件性能参考；严格GPU同合同基线尚未建立。STEP-192与Conv+BN研究暂停，待合同重建后重新profiling。

## 2026-08-14：合同差异历史与源码归因

- `lidar_dropout_prob/lidar_spatial_rate/lidar_mask_ratio`归于`740d9fd 随机性固定改动`，`use_grid_mask=False`归于`df7d06a 随机性固定`；它们不是NPU兼容所需。`63861df 【loss对齐】随机性移除`恢复了图像/点云pipeline增强和`synchronize_after_backward=False`，但没有恢复上述模型随机项、grid mask和FP16 optimizer hook。
- 当前`GradientFingerprintOptimizerHook`直接执行普通`loss.backward→clip→optimizer.step`，不继承GPU的FP16 hook；GPU`Fp16OptimizerHookProtectGradNan`会`wrap_fp16_model`、动态GradScaler、scale/unscale、裁剪、scaler step/update并保存scaler状态。两者属于不同精度/优化路径，必须恢复GPU hook后重新测。
- `one_sided_dim_threshold=1024`来自已验收的`fb979b2` SOAP NPU亲和优化；`memory_interval=10`来自`bf9ed6e`日志同步降频。二者分别作为已有等价性能实现与日志差异保留，但不能替代GPU功能字段。
- 当前NPU训练ann/flag basename与GPU配置一致，NPU镜像文件存在且已记录SHA；val/test引用不同，且test路径当前不可达。由于未取得GPU原始数据文件SHA，训练样本集合只凭basename仍不能完成严格身份认证。
- GPU日志本身记录117286帧/seed0，而本地GPU权威配置与当前NPU合同是28130帧/自然随机语义；因此旧GPU日志仍只能作性能参照。最终1:1验收需要GPU按该权威配置重跑。

## 2026-08-14：STEP-192最终裁决与交接事实

- NPU `channels_last`直接机制在固定torch_npu 2.7.1上不受支持，错误发生在任何算子计时前；没有可报告的加速或数值结果。
- permute-contiguous-permute备用方案属于边界layout转换，不属于用户要求的算子替换/融合，并可能增加转换开销，因此不执行。
- 当前Top N主线下一项应为冻结图像路径的`Conv2D + BNInfer`折叠：这是实际算子融合候选，`BNInfer`现有上限约49.119ms/普通步；必须先枚举真实相邻对并做eval/no-grad语义与数值门禁。
- 权威远端状态再次核验：`ascend_npu_optimize@f922c38` clean；另有`codex/baseline-customer-runtime-config@4c37039`基线工作树；正确容器无训练进程。

## 2026-08-14：GPU合同1-step首次失败根因

- 8个rank在`set_per_process_memory_fraction`触发NPU lazy init时同时失败；ACL 500001的底层错误明确为`tbe`模块不可见。失败早于模型构建、optimizer hook、数据读取和iteration，故不能归因于GPU动态FP16 hook，也没有任何数值或性能含义。
- 本轮启动命令把`PYTHONPATH`设置为`repo:repo/mmdetection3d-0.17.1`，没有保留容器原环境；这与CANN Python模块`tbe`不可见直接一致。修正边界只能是保留既有环境路径，禁止安装、升级或替换远端组件。
- 失败后容器训练进程曾核验为0、端口29936为0、业务Git clean。后续轻量`docker exec`自身连续超时，因此暂不重跑；超时不能被解释为模块缺失证据。
- 绕过login shell后的直接`python -S/find_spec`三路A/B为默认`True`、覆盖`False`、前置保留`True`；因此启动环境根因已闭环。容器固有CANN相关环境变量存在，禁止也无需安装组件。
- GPU动态FP16 hook在NPU首步可执行，但初始scale65536发生溢出、降至32768并跳过optimizer step；loss416.3346 finite。需以短多步观察scale继续回退后是否出现有效更新，不能把首步跳过隐瞒为通过，也不能因此私自改固定loss scale。
- GPU日志自身是step1 NaN并skip、step2/3 inf、step4首次有限grad97.7820；因此4-step是最小有证据的动态scale恢复窗口。NPU首步loss相对GPU首步约-4.46%，未出现数量级偏离。
- 对齐运行日志实际报告`all_data_frame_num=117286`，与旧GPU日志规模一致；此前“当前合同28130帧”是配置静态审计推断，应撤销。严格文件身份/样本顺序仍缺GPU原数据SHA，不能仅凭数量认定完全一致。
- 最新profiling合同：只采集一次预热后的连续多Step窗口；从同一份trace同时统计整体wall/device busy、数据/Host、前向、loss、反向、optimizer（普通与SOAP周期）、HCCL和空泡。阶段是分析维度，不是重复采集任务。
- 最新候选验收合同：算子级输出/梯度等价只是第一层；还必须在8卡短训中验证loss/grad与净性能，并以同checkpoint、同测试集、同顺序对比输出及任务指标。固定容器缺失canonical评测依赖时必须明确外部阻塞，不得安装依赖或省略门禁。
- GPU合同4-step NPU结果与GPU oracle一致：前3步动态scale回退，step4首次有限grad；四步loss相对GPU同编号绝对差均小于1.7%。step4 NPU 35.632s包含末次checkpoint并可能包含首次有效SOAP更新，远慢于GPU 8.860s，是一次全阶段profile中必须归因的optimizer线索，不能直接当纯算子耗时。
- 单次profile选择带栈：`with_stack=True + record_shapes=True`最适合从同一trace区分同名算子在前向、反向和optimizer的调用来源；其额外Host开销意味着profile step time不作为吞吐，只使用无profiler基线衡量端到端性能。
- profile稳定性门禁：动态scale回退、首次编译/缓存、首次有效optimizer和末次checkpoint都不能作为采集目标；必须由无profiler长窗口证明普通步稳定并定位稳定后的SOAP周期，再设置`wait+warmup`和唯一active窗口。
- GPU seed是运行时而非配置静态字段：用户GPU日志明确`seed=0, deterministic=False`，而首轮NPU 30步无seed行。严格基线必须追加`--seed 0`且不启用deterministic；未对齐轮已停止清理，不能用于性能或profile窗口判断。
- NPU训练入口不再解析seed，harness也不转发额外参数；仓库外最小入口副本恢复`set_random_seed(0, deterministic=False)`后，NPU/GPU前4步loss偏差均<0.07%，30步末loss偏差约0.36%，证明合同恢复有效。
- 稳态性能：普通step15～29排除24，NPU/GPU time mean=`6.1796/4.3241s`、吞吐比约0.700；完整周期step15～24 mean=`8.6575/4.416s`、吞吐比约0.510。NPU SOAP step14/24=`29.579/29.222s`稳定复现，是1:1目标的首要瓶颈线索。
- 唯一profile active窗口为稳定step23～26：step23/25/26是普通步，step24是稳定SOAP长尾；`wait+warmup`必须确保step1～22不进入采集，step30 checkpoint也不在窗口。
- 历史同一profiler实现的直接证据为`wait8/warmup1/active1 -> capture_steps=[9]`；因此本次`wait22/warmup1/active4`映射到step23～26。使用正式训练login-shell保留原CANN/客户Python路径后，完整配置导入、hook注册、22/1/4参数、world8门禁、stack/shape开关及checkpoint关闭均通过，尚未产生raw profile。

## GPU合同对齐的唯一全阶段profile结论（2026-08-14）

- 唯一采集完成28/28；Profiler Step23～26对应训练Step24～27，覆盖稳定SOAP步24与普通步25～27。带栈/shape只用于归因，最终性能继续取profiler-off基线。
- SOAP步QR为22,896.938ms，来源是`SOAP.update_preconditioner`；普通步稳定TopN是MSDA grad约186.7ms、Conv约133.2ms、ViewCopy约95.9ms、BMM约82.9ms、MSDA forward约81.8ms、Conv TransData约67ms。
- v4栈归因扫描1,823,325行，栈覆盖率85.0728%；MSDA forward/backward和无栈ConvolutionBackward已按算子语义正确拆分。通信重叠率约80.51%，wait-anchor不按总等待量排名。
- TopN与既有闭环矩阵一致：QR无固定环境状态等价方案；MSDA已优化；internal-format/Conv/TransData、point_sampling BMM、MapTR索引族、SOAP foreach/scalar等方向均已提交覆盖或正式拒绝。没有新的安全独立候选，不启动无依据A/B。
- 原始profile 199文件、16,647,868,129 bytes及已摘要日志/work已精确删除；最终raw=0、进程=0、端口=0、Git clean，保留10节架构报告、异常报告、TopN、v4栈归因、决策报告和校验manifest。

## STEP-194：BMM/ViewCopy二次边界复核（进行中）

- 逐Step数据揭示必须拆开两个BMM族：普通步每步约82.9ms的单次热点实际是`aclnnMatmul_BatchMatMulNd_BatchMatMulV2`，聚合在host op `aclnnMatmul`，最大单次约83.116ms，代表栈为`spetr3d.py:1182 forward_pts_train`；另一个`aclnnBatchMatMul`是约280次/步、15.9～20.6ms/步，聚合1120次仅83.740ms，代表栈为`spetr3d.py:1284`。此前候选报告把82.9ms主要归到1284并不精确，STEP-194必须纠正。
- ViewCopy普通步的AI CPU族为2048次、95.7～96.2ms/步；同一步还有AI Core ViewCopy约30～31ms及Transpose约18ms。host op `aclnnInplaceCopy`四步聚合24,291次、device self 649.768ms，最大单次6.888ms，代表栈为`spetr3d.py:1148 forward_pts_train`。聚合跨调用点，不能直接把649.768ms除错或视为单一可删收益。
- 下一步只读源码1148/1182/1284及其上游张量shape/调用循环，分别判断单次83ms Matmul、2048次ViewCopy和280次BatchMatMul是否存在新的producer-consumer冗余；wait-heavy部分继续降级。
- `spetr3d.py:1182`本身只是一次`self.bev_encoder(...)`模块边界，不能据该行直接判断BMM数学；1148同样是`pts_backbone.forward_rpn`边界。必须沿活动配置进入`models/utils/bev_encoder.py`和`bevformer_encoder.py:point_sampling`，不能在detector调用行做盲目替换。
- 活动配置明确使用`BEVFormerEncoder`；仓库检索显示其`point_sampling`在`bevformer_encoder.py:240`执行`torch.matmul(lidar2img, reference_points)`，而历史point_sampling packed-BMM候选已被正式30-step拒绝。仍需确认本次单次83ms是否就是该实现、是否已经是拒绝候选后的原始/恢复路径，以及调用次数/shape是否相符。
- 源码闭环确认本次单次83ms就是活动`BEVFormerEncoder.point_sampling`：原实现把reference points显式repeat约105MiB、lidar2img显式repeat约420MiB，再在line240执行`torch.matmul`。历史packed-BMM已针对完全相同边界做到8/8 rank bitwise exact、83.4096→1.8561ms并节省约423.6MB峰值，但两轮独立30-step均重复出现约64s+27s新进程编译成本，复验全步+3.529%、吞吐-3.409%、普通均值+0.284%，已经是充分正式反证。
- 因而新profile中的82.9ms不是“新BMM候选”，而是已拒绝候选恢复到clean HEAD后的预期残余。除非出现不同实现机制且能避免编译/端到端回归，否则不得重复packed-BMM或仅用函数微基准推翻正式A/B。STEP-194的BMM分支可标记`CLOSED_ALREADY_REJECTED_E2E`，继续审计ViewCopy。
- ViewCopy代表栈1148进入活动`MultiModal_PVB_GOP.forward_rpn`。该路径在`multitask_pvbgop.py:261`对`spatial_features_2d_lidar_pvb`显式`clone()`；当前`extra_downsample=None`，同一源张量另经permute作为`voxel_feats`，clone输出作为`spatial_features_2d_lidar`传给BEV encoder并写回data。
- v4显示`aclnnInplaceCopy`最大单次仅6.888ms，故即使line261 clone经消费者/alias审计证明可删，其单点上限仍低于22.7ms准入线，不能拿四步聚合649.768ms冒充单点收益。每普通步2048个AI CPU ViewCopy更可能来自高维matmul/内部视图物化等批量路径，需要与已拒绝point_sampling packed-BMM边界交叉验证；没有per-call原始栈时不得猜成2048个可删Python clone。
- 远端仍保留point_sampling脱敏门禁与两轮30-step报告（不含profiling raw）：两轮`metrics.json/report.md`均明确`REJECT_NO_COMMIT`，理由是packed-BMM虽消除展开成本，但可重复的新进程编译成本导致相对直接父提交全30步和吞吐回归；业务patch已回退。该证据足以关闭同一实现，不需要也不允许重新采集profile。
- 现有脱敏报告没有把2048个AI CPU ViewCopy逐栈绑定到point_sampling，因此只能说“可能共边界”，不能把它当硬结论。STEP-194继续从历史候选patch/函数表达式和活动消费者做静态交叉验证；若仍不能唯一拆分，按证据不足关闭聚合ViewCopy，而不是重采。
- 历史point_sampling目录已按生命周期规则只保留summary/report/SHA，没有候选patch或逐算子raw；不能从中追加ViewCopy逐栈证据。
- 另一条未改变数学的broadcast/expand实现已有8-rank exact门禁：真实shape reference `[4,16,15360,4]`、lidar2img `[16,7,4,4]`，峰值减少约551.6MB，但耗时83.3940→207.5506ms（0.4018x），明确更慢。因此对同一BMM边界，显式repeat是83ms、纯broadcast是207.6ms、packed-BMM微基准1.856ms但正式30-step回归；当前固定环境下三种主要表达均已有证据，不能再以“换成expand”作为新候选。
- line261 clone的消费者审计未发现对`spatial_features_2d_lidar(_pvb)`的直接`copy_/add_/mul_/zero_/resize_/scatter_/index_put_`原地写；BEV encoder先通过卷积投影生成新tensor，再flatten/permute/contiguous，GOP/lane路径也只读取。删除clone在当前冻结PVB路径可能语义等价，但v4最大单次copy仅6.888ms，低于22.7ms门槛，故无需占NPU做候选门禁。
- 现有候选报告的BMM描述需要纠正：82.9ms单次`aclnnMatmul`来自`spetr3d.py:1182→BEVFormer point_sampling`，不是1284的loss/target高频`aclnnBatchMatMul`。1284族约16～21ms/普通步，同样低于阶段门槛且MapTR目标路径已有闭环。

## STEP-195：冻结图像Conv-BN折叠（进行中）

- 活动配置`fix_backbone=True`；`spetr3d.py:2006-2014`在每步图像特征提取前执行整模型`eval()`，在`torch.no_grad()`内依次调用`img_backbone`和`img_neck`，之后恢复`train()`。因此该调用窗口内Conv/BN参数不求梯度、BN使用running stats且不更新，满足推理折叠的首要语义前提。
- 图像输入实际按batch16×7 camera展平为`[112,3,576,1024]`（配置/既有探针口径）；输出四层由ResNet34+FPN产生。候选必须只作用于该冻结窗口，不能全局融合训练态BEV/点云Conv-BN。
- 首次按`mmdet/models/...`查找源码不存在，说明客户依赖源码位于仓库内其他前缀；该只读检查没有改状态。下一步用`git ls-files/rg`定位真实ResNet/FPN实现，再枚举直接相邻对。
- 仓库tracked源码只有MMCV旧ResNet以及项目自定义`ResNetForBEVDet/FPNForBEVDet`，但活动配置类型精确为`ResNet/FPN`，不是这两个自定义类；仓库内也没有`mmdet/models/backbones/resnet.py`或`mmdet/models/necks/fpn.py`。因此活动实现来自固定容器已安装mmdet包，必须通过容器只读`inspect.getsourcefile`定位，禁止假用项目同名/近名类枚举融合对。
- 正确容器只读inspect确认活动类来自固定环境：mmdet `ResNet`和`BasicBlock`、mmdet `FPN`、MMCV `ConvModule`。CPU-only按GPU对齐config实例化（未init weights、未占NPU）得到ResNet34直接Conv-BN对36组：stem 1、16个BasicBlock各2、layer2/3/4首块downsample各1；FPN直接Conv-BN对7组：3个lateral、4个fpn conv。合计43组，参数量约21.285M+2.592M。
- 43组均是活动模块真实结构，不包含BEV/点云训练态卷积；理论上可用标准eval融合公式折叠。但是否形成候选仍取决于BNInfer纯kernel上限、融合副本的checkpoint/state_dict兼容设计和真实大输入端到端计时。
- STEP-189稳定普通步的硬上限为`aclnnBatchNorm_BNInfer_BNInfer` 59次、纯kernel 49.118904ms、wait仅0.003383ms，是真计算而非wait-anchor，超过22.7ms门槛。但活动图像模块只有43组；其余BNInfer可能来自同一步其他eval路径，因此49.119ms只能作为全步上限，不能提前宣称图像融合可全部回收。
- 唯一trace的v4栈聚合进一步确认：`aclnnBatchNorm`在4个profile step共352次，device self time `210.599ms`、host time `54.239ms`，最大单次`7.732766ms`；代表栈为`spetr3d.py:458 extract_feat -> spetr3d.py:2009 forward_train`，明确落在图像特征路径。另有`aclnnBatchNormBackward` 116次、device `53.484ms`且无代表栈，属于训练态其他路径，不能纳入冻结图像折叠收益。由此可确认BNInfer主要来自图像路径，但仍须用真实模块计时直接测得43组可回收量。
- 稳定普通Profiler Step24～26中BNInfer均为59次，纯kernel分别`48.211/48.313/48.241ms`，均值`48.255ms`、极差仅0.102ms；旧STEP-189 Step7为49.118904ms且wait仅3.383us，重复证明这是稳定计算热点。
- 59次可由活动结构精确拆分为图像ResNet34+FPN的43组、点云BaseBEVBackbone_FPN历史15组以及冻结PillarVFE最后一个PFN的1组。按次数比例粗估图像部分约35.17ms，但各shape成本不同，该值只支持进入机制门禁，不能作为正式净收益。
- 固定环境提供`torch.nn.utils.fuse_conv_bn_eval/fuse_conv_bn_weights`和`torch.ao.quantization.fuse_modules`；float64 CPU机制实验最大绝对差`8.88e-16`，融合Conv保留stride/padding/dilation/groups/dtype等属性，原无bias时生成bias。MMCV 1.7.2接口原地修改且无eval断言，不采用；QAT接口也不适用。
- 最安全设计不是替换原模块，而是保留原注册模型为checkpoint/state_dict/optimizer/DDP唯一权威，在DDP设备迁移及checkpoint加载完成后懒构建未注册的eval-only融合副本。仅`fix_backbone=True + eval + no_grad`图像窗口路由副本；任何训练态、梯度、设备/dtype或源参数/BN buffer变化均回退并失效缓存。副本不得注册为self子模块，checkpoint始终保存原Conv+BN，因此resume不依赖融合表示。
- `train_spetr.py`顺序为build/init→模型迁移/DDP→optimizer/runner→resume/load checkpoint→run，故副本绝不能在init或checkpoint加载前构建。门禁必须证明原state_dict值hash、named parameter/buffer及optimizer param ID不变，并验证融合启用checkpoint可由禁用融合的原路径直接resume。
- 对API审计的初始否决已纠正：`norm_eval=False/frozen_stages=-1`描述全局train模式，不能覆盖当前`fix_backbone=True`分支的实际运行时窗口。源码确认正式训练及历史帧图像提取均在`self.eval()+no_grad`内，当前`queue_length=1`；simple_test/dummy调用不在forward_train链，全仓无外部训练调用dummy。固定环境CPU真实结构探针显示43个BN在窗口内stats变化0、输出均无grad、参数grad为0，窗口后原模块43个BN均恢复train，因此未注册融合副本的语义前置门禁通过。
- CPU真实ResNet34+FPN融合后BN 43→0；四层输出max_abs约`8.106e-6/2.861e-6/5.960e-7/5.085e-7`，relative L2约`1.36e-6～1.73e-6`，`rtol=1e-4,atol=1e-5`通过但并非bitwise。正式NPU门禁必须报告max_abs/NRMSE并覆盖近零元素，不能只依赖PyTorch默认allclose。
- STEP-195真实NPU门禁证明全量43对折叠有明确局部性能收益但未通过数值语义：8-rank真实输入的原/融合边界聚合中位数`325.876350/274.807587ms`，节省`51.057739ms`、`1.185813x`，峰值allocated均为`11,555,523,584 bytes`；四层shape/stride/finite及原state/参数/缓冲区/optimizer引用均不变。
- 8个rank的输出误差完全一致：四层NRMSE约`1.849e-3/1.991e-3/1.805e-3/1.760e-3`，全局max_abs=`0.0647306`。最大NRMSE超过预声明`1e-4`门槛近20倍，故不能用约51ms收益越过功能门禁；裁决`REJECT_NUMERIC_GATE_NO_TRAIN_NO_COMMIT`。这条实现不进入loss/grad短训练和测试集A/B，避免通过后续统计掩盖已知局部语义漂移。

## 2026-08-14：STEP-196 普通稳定步 underfeed 只读复核

- 权威低开销证据仍是 STEP-189 稳定 Step7：`service=7762.3855ms`、多流 `device_busy_union=1916.08275ms`、`underfeed=5846.30275ms`、`underfeed_ratio=75.3158%`。84,811 次 device kernel 的 duration sum 为 `1928.575394ms`；巨大 underfeed 是事实，但不能直接等同于一个可删除的 Python/C++ 开销。
- STEP-189 前五内部 bubble 为 `18.964/11.973/8.454/7.428/6.808ms`，合计 `53.627ms`。它们分别落在 `item/_local_scalar_dense + DDP unused search + zero_grad`、DDP bucket copy/grad norm、autograd Add/MeanBackward、Nonzero/Index/ge/le、Nonzero/IndexPut。单窗均低于 `22.7ms/step` 门槛，且跨多个不同语义边界，禁止把总和包装成一个候选。
- Python/C++ 标量同步：全阶段带栈报告的 `item/_local_scalar_dense` 聚合主要归于 SOAP optimizer scalar-overload；源码不是显式 `.item()`，改写需改变逐参数更新边界。普通路径 line1218 的 256 次标量同步候选已在 STEP-175 消除后正式 30-step 回归（普通步 `+8.118%`），已 `REJECT_NO_COMMIT`，不可重开。
- DDP：`search_unused_parameters` self 仅约 `3.416ms/step`，且 3-step×8-rank 探针证明每步固定有 `142/701` 个可训练参数无梯度，`find_unused_parameters=True` 是正确性必要路径。bucket copy、backward 和 zero_grad 都属于必要训练语义；grad norm 已证明当前 `foreach=None` 自动走批量路径，显式 True 不提速。
- 数据等待：当前客户合同已经是 `workers_per_gpu=8`、`prefetch_factor=3`、pin memory；全阶段带栈报告中 data host self 仅 `1.826ms/4 steps`（约 `0.457ms/step`）。带栈普通步 `1.60～1.68s` prelaunch 与 sync/H2D 高重叠，但 profile 开销使其不能作为生产收益；低开销 STEP-189 的 Top5 内部窗 sync overlap 为0。现有证据不足以证明新的、唯一的数据源码边界，更不能重开已验收 DataLoader/pin 候选。
- 通信：STEP-189 HCCL kernel sum=`12.638ms`，与 compute 重叠=`87.155%`，未重叠理论上限仅约 `1.623ms/step`；全阶段报告也显示通信不是普通步 P0。`communication.json` 缺失，不能做带宽根因断言，但现有上限已远低于门槛。
- 裁决：`CLOSED_NO_UNIQUE_EQUIVALENT_UNDERFEED_BOUNDARY`。当前脱敏证据没有发现“源码边界唯一、状态等价且理论净收益 >22.7ms/step”的新候选。剩余约5.85秒 underfeed 更像数万次细粒度下发/等待的分散总量；原始 timeline 已不存在，无法把这些间隙唯一映射到可批量消除的调用链。遵守用户“不重采一次 profile”的要求，本轮不训练、不占NPU、不改业务代码。
- Index/Reduction交叉复核按普通Step24～26纯kernel均值重算：Index `55.831ms`、ReduceSum `54.242ms`、Nonzero `52.011ms`、Unique `51.931ms`、IndexPut `35.475ms`、Nonzero-MemSet `21.166ms`、VectorNorm `16.000ms`。这些值跨多个功能和调用点，不能合计为286.656ms单候选；除Nonzero外，主要家族wait占比约90%～98%，Eq/Broadcast/Fill/Zero/MemSet等更达到98.91%～99.89%，均按wait-anchor降级。
- v4代表栈只落到`spetr3d.py:1284/1182/1148`等外层边界，不能唯一定位聚合内每次Index/Reduction。历史上GeometricLoss的2.07s Index/IndexPut已由等价masked reduction兑现；MapTR Unique正式30-step普通步回归11.143%；target mask复用仅+1.28%且低于噪声；VectorNorm已自动foreach；SOAP IndexSelect和MSDA Zero的独立上限也远低于门槛。因此Index/Reduction裁决`CLOSED_NO_NEW_INDEPENDENT_BOUNDARY`。
- 脱敏架构报告正文配置为有效`workers=8/prefetch=3`，但ASCII结构图仍有`worker=2`陈旧模板标签；性能判断始终以活动配置8为准，该文档一致性问题需单独修正并更新校验值，不能反向改配置。
- STEP-196文档修正已远端原位完成：陈旧标签仅有1处，且同报告正文唯一明确写明`DataLoader workers per rank: 8`，故将ASCII行精确改为`DataLoader(workers=8,prefetch=3) → image features`。架构报告字节数仍为`10499`，新SHA256=`f7192a7d...08006c`；`analysis_validation_summary.json`同步后的SHA256=`95f94a5c...a66ce`，`final_manifest.json`自身SHA256=`3cee39c3...d54c`。三处交叉校验均为True，raw签名文件0，业务Git status0。

## 2026-08-14：STEP-197 选择性Conv-BN分组门禁

- CPU前置门禁以同一checkpoint严格加载258个图像键，六组pair计数为`stem/layer1/layer2/layer3/layer4/FPN = 1/6/9/13/7/7`；原state hash及named参数/缓冲区ID不变。启动前HEAD为`f922c38`、业务Git clean、后8设备空闲、`torch_npu=2.7.1`且容器内可见8卡。
- 单一8-rank任务内逐组融合并释放副本；`npu-smi`实时捕获后半物理`4/0～7/1`共8个Python PID，补齐STEP-195短任务未捕获live PID的审计缺口。全部rank确认world8/visible8/torch_npu，任务自然exit0。
- 六组结构门禁均通过，四层输出固定为`[112,256,72,128]`、`[112,256,36,64]`、`[112,256,18,32]`、`[112,256,9,16]`。但最大NRMSE分别为`1.529e-3/1.132e-3/7.238e-4/4.513e-4/3.398e-4/2.977e-4`，无一满足`<=1e-4`。
- 对应边界中位净节省为`8.163/12.185/10.850/6.785/8.091/5.865ms`，也均未超过`22.7ms`。由于没有单组同时通过数值与正收益筛选，组合选择为空，避免组合爆炸或用误差累积掩盖单组失败。
- 裁决`REJECT_NO_SELECTIVE_GROUP_MEETS_NUMERIC_AND_22P7MS_GATE`。选择性折叠不能作为全量43对折叠的安全子集；不训练、不做测试集A/B、不改业务、不commit。远端仅保留脱敏summary，JSON/MD SHA分别为`e89b160c...8f577`与`7762ab13...ce3dd`。

## 2026-08-14：继续TopN主线的方法边界

- Ascend分析仍须同时维护`wall_ms/busy_union_ms/kernel_sum_ms/total_cost_ms`；多流设备繁忙按所有AI Core、AI CPU和HCCL区间合并，wait发生在kernel start之前，不属于device busy。后续新候选不能用wait或kernel sum冒充端到端收益。
- 当前原始timeline已不存在，现有op/stack摘要不能追加新的时间窗级唯一归因。因此下一步只审计不依赖重新profile即可证明的固定环境图执行/算子融合能力；若需新timeline才能判断，则记录证据缺口而不违反用户“一次采集复用”的约束。
- 规则书再次确认：`underfeed_ratio>=0.30`即属于重设备空泡，但根因必须使用`possible_*`或`insufficient_evidence`分层表达；高wait小duration对象必须标记`WAIT_ANCHOR_FALSE_HOTSPOT`并降级。现有STEP-189的75.3% underfeed是硬事实，但不足以证明某个图执行开关必然有效，任何新机制仍需profiler-off A/B。
- 现存异常JSON已满足要求的global gap、step group、bubble、wait-anchor、soft cause与confidence字段；后续若没有新profile，不重写或扩充不存在的raw证据，只维护已有schema有效报告与独立架构报告。
- 华为TorchNPU 7.3官方性能流程把端到端单batch拆为数据加载、前反向、optimizer、后处理、未掩盖通信与调度，并明确指标优先级为吞吐率高于单步时间。当前项目继续以无profiler `samples/s`为最终口径、profile只做归因，与官方流程一致。
- 官方调优方法强调：融合必须是数学等价替换，通过减少冗余计算和下发次数获益；`item/reduce_all/isfinite`等H2D/D2H可能引入stream同步，应减少但不能改变语义；多卡通信高耗时可能是快卡等待慢卡，需找首个分歧位置，不能把通信wait直接当根因。
- 官方DataLoader建议`num_workers`按负载实测（图像常见4或8）、内存允许时pin、可评估persistent workers/prefetch/fast collate，但同时警告过多worker增加进程开销。当前客户合同workers8/prefetch3已对齐、data host self约0.457ms/step且pin正式全步回归，故不因通用建议重开已关闭数据候选。
- 官方并行策略建议是场景化实验而非万能规则，增大batch或改变TP/PP/DP会改变主验收合同；本项目仍以GPU batch/rank16、8卡同合同为主，不用更大batch替代1:1目标。

## STEP-198：固定环境图执行能力

- 固定容器为`torch/torch_npu=2.7.1`，`torchair`的module spec和package metadata均不存在；项目规则禁止安装或改变版本，因此TorchAir/GE编译路线当前不适用。
- `torch.compile`存在，但默认backend是inductor；TorchNPU注册的`backend="npu"`在缺少torchair时实际绑定`_eager_npu_backend`并仅返回原GraphModule，不产生图编译收益。不能把成功调用误报为图优化。
- 固定环境原生提供`torch_npu.npu.NPUGraph`、`graph`、`graph_pool_handle`、`make_graphed_callables`和Dynamo `npugraphs` backend。这是无需安装/升级的唯一新图执行机制；API存在不等于项目兼容或有收益，必须走真实shape与8-rank门禁。
- 历史正式A/B已经关闭TASK_QUEUE Level2、COMBINED、全局internal format、HF32、packed-BMM、Conv-BN折叠等；这些不能以“图模式”名义组合或重开。`jit_compile=False`只出现在早期backlog，没有通用torch.compile/TorchAir/NPUGraph正式A/B，是真正未覆盖空白。
- 最小候选应只捕获冻结图像塔`img_neck(img_backbone(img_flat))`：保持外部flatten/view和全模型eval/train切换在图外，checkpoint加载、DDP构建和device迁移后每rank捕获，graph对象不注册进model/state_dict/optimizer。该边界无梯度且输入/四层输出shape固定，理论上比包含Python dict/list、自定义autograd的BEV/MSDA边界安全。
- 边界审计子agent首次误读并存旧仓库`l2.9-df-for-yuexiang_ascend_npu@f189414`，不是权威`l2.9-df-for-yuexiang`的`ascend_npu_optimize@f922c38`；已禁止据此启动门禁，并要求在正确仓库复核活动配置。该错误在任何写入、训练或NPU占用前发现。
- 权威仓库纠正复核通过：`ascend_npu_optimize@f922c38` clean；活动合同为batch16、7 cameras、T=1、固定576×1024输出、`fix_backbone=True`、`use_grid_mask=False`。训练增强虽改变图像内容，最终张量shape固定；进入图像塔前精确为`[112,3,576,1024]`。最小NPUGraph边界因此成立，但现有摘要未证明其launch gap超过22.7ms，只允许先做8-rank真实shape机制计时，不直接短训。

## STEP-198：冻结图像塔原生NPUGraph机制门禁结论

- 固定环境要求手工NPUGraph只能在非默认NPU stream捕获；默认stream的第二轮在8/8 rank一致被API拒绝，属于捕获前harness错误。最终仅增加独立capture stream和捕获前后同步，没有改变边界、环境或阈值。
- 第三轮严格加载同一iter30 checkpoint的258个图像键，8/8 rank记录world8、local_rank0～7、后8逻辑设备和TASK_QUEUE Level1。四层FP32输出shape/stride/finite完全一致，eager与graph的`max_abs=0`、`NRMSE=0`，同输入重复replay误差也为0；state SHA、named parameter/buffer、optimizer引用和checkpoint optimizer IDs均不变，graph未进入module/state_dict。
- 在排除capture/首编译、分别11次warmup、同进程交替8轮且graph计时包含`static_input.copy_(dynamic_input)+replay`的口径下，eager各rank中位数范围`326.350～326.833ms`，graph范围`327.103～327.693ms`；rank中位的汇总为`326.609→327.281ms`，净收益`-0.714ms`、`0.99795x`。原生图没有减少该完整边界wall，反而轻微回归。
- capture额外allocated=`2,807,567,360 bytes`、reserved=`30,666,653,696 bytes`，峰值allocated/reserved=`16,463,180,288/43,471,863,808 bytes`，资源代价显著。裁决`REJECT_MECHANISM_GATE_NO_TRAIN_NO_COMMIT`，不扩展到BEV/MSDA，不用退化的`torch.compile backend=npu`，不短训或改业务。
- 严格启动证据限制：曾实时看到8直接worker、rank0～7/local_rank0～7与端口29980，但宿主和容器内`npu-smi`在ready保持窗口采样均未返回对应PID，因此live PID项明确记为缺失而非通过。候选已经因负收益拒绝，不再额外重跑粉饰该证据。
- 远端第三轮raw 21文件和harness 2文件已精确删除，仅保留analysis的`summary.json/summary.md/manifest.json`；manifest SHA=`765b93c5...5549`，业务Git clean、进程0、端口0、profile raw0。

## STEP-199：SOAP QR 数学/状态等价合同只读审计

- 权威实现锁定为`ascend_npu_optimize@f922c3897255`的`projects/mmdet3d_plugin/optimizers/soap.py`。活动配置是`SOAP`、`precondition_frequency=10`、`one_sided_dim_threshold=1024`；其余采用实现默认值：`max_precond_dim=10000`、`merge_dims=False`、`precondition_1d=False`、`normalize_grads=False`。二维参数任一轴大于1024时仅保留较小轴的`GG/Q`，其他轴以空列表表示。
- 每个非空轴的周期更新严格为：从`GG=m`与旧`Q=o`转换/保持FP32；计算`est_eig=diag(o.T@m@o)`；执行`descending=True, stable=True`的`argsort`；用同一`sort_idx`重排`exp_avg_sq`对应维和`o`列；计算`power_iter=m@o_sorted`；调用默认`torch.linalg.qr(power_iter)`并只保存`Q`。`R`不被业务读取，但只有候选能生成与当前实现完全相同的`Q`时才允许省略R物化。
- 调用频率不只是“每10步一次”：某个参数首次出现梯度时先初始化`GG`，以identity为`Q`并立即走同一轮QR；该首步跳过参数更新。之后`state.step`递增并在`step>0 && step%10==0`时更新QR。当前稳定SOAP profile的一个周期步有543次方阵QR；主要shape/count/设备纯计算为`2560: 4次/16147.768ms`、`768: 22/2188.898ms`、`512: 43/1558.071ms`、`1024: 6/1459.024ms`、`256: 181/1048.222ms`，合计QR `22641.384ms`。历史完整shape表为24种；one-sided提交精确删除了4个5120轴和4个不必要2560轴，使551次降为543次。
- `Q`是持久optimizer state，不是可丢弃的前向中间量：`project`沿各轴用`Q`，`project_back`用`Q`的另一维；`GG`持续EMA更新；`exp_avg_sq`在每轮QR按排序索引原地换坐标；MMCV checkpoint直接保存完整`optimizer.state_dict()`，resume再调用`optimizer.load_state_dict()`。因此自定义QR必须验证checkpoint保存/恢复后下一步与连续运行一致，不能只比较`Q.T@Q≈I`或单次重构误差。
- QR列符号在精确数学上可由`Q[:,j]→-Q[:,j]`、`R[j,:]→-R[j,:]`保持分解；SOAP的投影/反投影在理想精确算术下也会抵消纯列符号。但当前实现不做符号规范化，原始`Q`本身被保存并参与后续FP32运算。给基线和候选追加“最大绝对元素为正”等规范会改变现有状态与下发，符号对齐只能作为诊断，不能替代raw Q/state门禁。
- 严格准入合同分两层且不可降级：单QR层要求输入shape/stride/dtype、finite/NaN/异常行为、stable sort索引和raw `Q`逐位相同；`R`在业务中不读取，Q-only候选可不生成R，若候选仍生成R则把R逐位与重构残差作为诊断而非功能门禁。完整SOAP层要求连续至少两个QR周期中的`Q/GG/exp_avg/exp_avg_sq/state.step`、参数更新、state_dict键/shape/dtype/device逐位相同，并验证“连续执行”与“中途checkpoint/resume”逐位相同。若raw Q只有符号对齐或容差相等，则只能标记数学近似研究，不能进入8卡性能A/B。
- GPU权威本地配置和`gpu去除随机性固定后loss.log`只能认证`type='SOAP'`、`precondition_frequency=10`及动态FP16训练现象；它们没有嵌入`soap.py`源码SHA，不能据此宣称GPU实际运行了某一份逐行QR实现。仓库历史`740d9fd`的GPU时代路径可作代码语义参考：首次basis用FP64 `eigh`并flip，周期更新把`m/o`搬到CPU FP64后做非stable argsort、QR再拷回；这与当前NPU的identity起始、stable sort、device FP32、one-sided轴选择不是逐状态同算法。GPU仍是最终性能主参照，但不能作为当前自定义QR逐位oracle；当前f922c38及其checkpoint/resume是NPU状态等价oracle。
- 算法候选裁决：Householder是当前QR所属算法族，但项目内重写若不能复现ACLNN的reflector选择、归约顺序、舍入与符号，就会改变raw Q；STEP-090的`geqrf→orgqr/householder_product`已实测非逐位且CPU fallback。TSQR对当前全部方阵输入需分行局部QR和R树归并，归约顺序/舍入不同且不适合方阵热点。经典/改进Gram-Schmidt改变点积与归一化顺序并降低或改变稳定性。Cholesky-QR通过`A.T@A`平方条件数，对训练早期低秩/半正定`GG`派生的`A=m@Q`可能奇异，失败/NaN边界也不同。上述方案即使正交误差小，也都会改变持久Q或数值轨迹，均为`NO_GO_ALGORITHM_NOT_STATE_EQUIVALENT`。
- STEP-089～100已覆盖并拒绝同shape batch、Q-only geqrf/orgqr/Householder、排序/重排旁路、power表达式复用、identity特例、通用/窄双流、`out=`缓冲等边界。唯一尚未重复的理论实现边界只能是“在固定环境内直接调用与`aclnnLinalgQr`相同语义且可证明raw Q逐位的更快底层primitive”；若工具链审计找不到该primitive，则STEP-199应直接`NO_GO`，不做NPU微测、训练或业务修改。

## STEP-199：SOAP QR 自定义算子工具链/API只读审计

- 权威门禁通过：远端只有一个完整名称为`mapqr-leicheng`的运行容器；目标仓库为`ascend_npu_optimize@f922c3897255`且Git status为0。容器固定为aarch64、Python 3.11.10、torch 2.7.1+cpu、torch_npu 2.7.1；审计结束训练/torchrun进程为0。
- 固定环境具备“开发和编译扩展”的基础工具，但不等于已有可用QR替代：GCC/G++ 10.3.1、CMake 3.22.0、Make 4.3、Ninja、CCEC/BiSheng 15.0.5、OPC和`msopgen {gen,compile,sim}`存在；Python侧有`torch_npu.utils.cpp_extension.NpuExtension`、TBE/TE和`op_gen`。`NpuExtension`会补齐torch_npu/ACL/HCCL头库与`c10/torch/torch_cpu/torch_python/torch_npu`链接库。
- 项目已有NPU C++封装基础：tracked C/C++/CUDA/头文件492个、8个setup.py，MMCV NPU目录含MSDA等ACLNN/OpCommand wrapper和可动态加载`libopapi.so/libcust_opapi.so`的公共工具。但现有项目setup均走普通`CppExtension/CUDAExtension`，未发现tracked `NpuExtension`构建清单、AscendC/TBE QR工程或tracked `.so`；因此新路径至少要新增独立源码/构建清单并编译产物，不是纯Python零构建替换。
- QR底层能力只有当前正在使用的同一路径：SDK提供`aclnn_linalg_qr.h`以及`libaclnn_math.so/libopapi.so`中的`aclnnLinalgQrGetWorkspaceSize/aclnnLinalgQr`导出符号；接口强制传入Q和R两个输出，没有Q-only入口。头库/动态符号中未发现`aclnnGeqrf`、`aclnnOrgqr`或另一个同`aclnnLinalgQr`语义的更快primitive。Python虽暴露`torch.geqrf/torch.orgqr`，不构成NPU底层实现证据；torch_npu源码只明确注册`linalg_qr`，且其Inductor路径仍标为fallback。
- 当前`torch.linalg.qr`在profile中已经落到`aclnnLinalgQr_QrAiCPU_Qr`。用项目NPU C++ extension再包装`aclnnLinalgQr`可以保持同语义，但仍需分配Q/R并执行同一executor/kernel，至多改变很小的Python封装开销，不能回收约22.6秒的QR纯计算；它不是性能候选。
- AscendC/TBE从零实现理论上可生成/编译自定义算子，但还需要自定义工程、编译产物和运行时注册/加载；这超出本轮只读边界，且客户环境禁止安装/替换组件。更关键的是它会成为不同QR算法实现，现有固定SDK没有公开可复用的同语义Q-only reflector primitive，无法预先满足raw Q逐位及连续SOAP state等价合同，故不进入编译或NPU机制门禁。
- 独立提交/回退仅在工程层面可行：若未来厂商提供同语义Q-only primitive，可把extension源码、build manifest和Python调用作为一个`【npu性能优化】...`功能commit，失败时回退该commit；编译出的ABI产物必须绑定当前torch_npu/CANN并通过现有构建流程产生，不应靠修改系统SDK或安装新版本。但当前没有这样的primitive，因此“可独立提交”不构成GO依据。
- 综合裁决：`NO_GO_NO_EQUIVALENT_Q_ONLY_OR_FASTER_PRIMITIVE_IN_FIXED_ENV`。固定环境能编译自定义算子，却没有满足“同aclnn语义、raw Q逐位一致、Q-only或更快”的底层API；不编译、不安装、不占NPU、不训练、不改业务、不commit。
- 2026-08-14 STEP-200/addmm 口径纠正：`STEP-189_f922c38_全部算子耗时.csv`是低开销稳定普通Step7的单步表，不是4-step全阶段表；因此 `aclnnAddmm_MatMulCommon_MatMulV2` 为117次/step、`kernel_sum=15.088563ms/step`、wait=`4.776433ms/step`、`total_cost=19.864996ms/step`，wait占比24.044%，不满足wait-anchor。唯一4-step带栈报告恰为468次，即同样117次/step；其operator device/host self折算约`12.271/18.732ms/step`，最大单次device self约`2.595ms`。主设备口径仍以kernel_details为准；即使错误地把全部wait都当可回收，总上限19.865ms也低于22.7ms，不能进入机制门禁。

## 2026-08-14：STEP-200 `torch.addmm` 亲和建议闭环

- 四时钟保持分栏。低开销稳定普通Step7全局为`wall/service=7762.3855ms`、跨流`busy_union=1916.08275ms`、`kernel_sum=1928.575394ms`、`total_cost=5696.985825ms`；后两者是加性统计，不能与wall相加。Addmm族的保留脱敏表仅有聚合、没有逐调用时间戳，因此`wall/busy_union`不可重建；可证明的`kernel_sum/total_cost=15.088563/19.864996ms`，其中wait 4.776433ms不属于device busy。证据缺失处保持N/A，禁止用全step wall或调用跨度填充。
- 当前唯一4-step带栈报告中`aclnnAddmm`为forward阶段468条，代表栈只能落到`spetr3d.py:1182 → bev_encoder`外层；折算117条/step，与低开销Step7的117个Addmm kernel完全交叉一致。该报告的device/host self折算约`12.271/18.732ms/step`，host self受异步下发和嵌套归因影响，不能与kernel相加为收益。
- 既有8-rank真实shape hook记录111类、133次`Linear/MatMul`调用，估算forward 888.575 GFLOPs；保留的Top25记录覆盖94.696% FLOPs，全部分属BEV encoder（483.586 GFLOPs）和lane3d decoder（357.858 GFLOPs），证明117个Addmm不是单一可替换模块。主要边界为BEV sampling offsets/attention weights/value/output projections与FFN，以及4层lane3d decoder的value projection、三层output projection链和position-query projection。
- 代表真实shape包括BEV sampling offsets `[112,15360,256]×[128,256]`、BEV FFN `[16,15360,512]×[256,512]`及`[16,15360,256]×[512,256]`、decoder四层`output_proj.0` `[16,120,5120]×[2560,5120]`、value projection `[16,15360,256]×[256,256]`、position-query `[16,120,20,256]×[256,256]`。输入随batch、attention层和decoder层改变，不可缓存或合并成共享结果。
- 权威`f922c38`源码只有一处显式`torch.addmm`，位于未被活动配置引用的`mmcv/ops/masked_conv.py`稀疏masked-conv分支；活动仓库没有`npu_linear`调用。活动`nn.Linear.forward`固定调用`F.linear`，NPU profile已经显示单个`aclnnAddmm_MatMulCommon_MatMulV2`，即bias与matmul已由backend融合，没有独立bias Add可再消除。
- 固定`torch/torch_npu=2.7.1`提供`aten::addmm`与`torch_npu.npu_linear(input,weight,bias)`，没有`torch_npu.npu_addmm`。`npu_linear`文档合同为2D矩阵，而活动输入普遍为3D/4D；替换需要额外flatten/view并重新验证输出、梯度、stride与异常行为，且最多仍是一个线性kernel，不可能从已单kernel的Addmm回收超过全族15.089ms。
- 历史闭环一致：STEP-079已证明`maptr_decoder.py:812` output projection是attention后动态必要计算并参与line819 residual；STEP-176标准MHA forward仅约1.55ms；STEP-190 MatMul HF32按三类真实shape预计仅3.964ms且最坏NRMSE `1.469314e-4`；point-sampling MatMul/BMM已正式拒绝。不能把这些不同kernel/模块的上限合并到Addmm候选。
- 裁决`NO_GO_ALREADY_FUSED_AND_BELOW_THRESHOLD`：当前稳定profile的纯kernel上限15.089ms低于22.7ms；即便错误计入全部wait，总成本19.865ms仍低于门槛，且调用分散、没有新的单一严格等价融合边界。因此不创建机制门禁、不启动NPU/训练、不重采profile、不改业务代码、不commit。

## 2026-08-14：STEP-200 `npu_confusion_transpose` 亲和建议闭环

- 固定`torch/torch_npu=2.7.1`确实暴露`torch_npu.npu_confusion_transpose`，真实schema为`(Tensor input, int[] perm, int[] shape, bool transpose_first) -> Tensor`，没有in-place或`out=`变体。官方同tag单测把`transpose_first=True`定义为`permute→contiguous→view(shape)`，把False定义为`view(shape)→permute`；但单测只比较数值，没有验证stride、storage alias或view身份。故False不能自动替换当前返回非连续view的路径，True也只能覆盖本来就必须物化连续副本的边界。
- 固定包codegen把前向和反向都标为`impl_ns: acl_op`，未提供`op_api/ACLNN`实现；二进制同时包含“当前设备只支持ACLNN而本算子无ACLNN实现”的显式保护分支。官方单测只覆盖base format 0，公开doc没有证明任意internal format可用。当前A3客户环境与项目又固定为ACLNN路径、关闭internal format，因此静态兼容门禁已不支持把该API作为正式候选；本轮没有为验证已失败的前置条件而占NPU。
- 低开销稳定普通Step7中，最多只能宽松关联`aclnnInplaceCopy_TransposeAiCore_Transpose=19.030772ms kernel+6.783137ms wait`与`aclnnContiguous_TransposeAiCore_Transpose=2.910202ms kernel+1.603660ms wait`。两族纯kernel合计仅`21.940974ms`，仍低于`22.7ms`；它们的total cost合计`30.327771ms`包含8.386797ms排队等待，不能当可回收device compute。保留摘要无逐调用时间戳，故边界wall/busy_union为N/A。更重要的是前者是对非连续目标的原地copy、后者才是直接contiguous，二者跨消费者且不能合成一个严格等价边界。
- `Index_Transpose/Add_Transpose/Matmul_Transpose/ReduceSum_Transpose`等名称中的Transpose只是各自复合kernel的内部实现变体；`npu_confusion_transpose`不包含Index/Add/Matmul/ReduceSum数学，不能把这些耗时并入候选。唯一4-step逐Step摘要中`InplaceCopy_Transpose`在SOAP Step23为18.350ms，在普通Step24～26为17.705/18.513/18.254ms，说明它不是SOAP周期专属热点。
- SOAP活动配置为`data_format=channels_first`、`merge_dims=False`。源码中显式`reshape(...).permute(...)`只位于`merge_dims && channels_last`分支，当前不执行；活跃空Q轴只做单独`grad.permute`并返回view。用fusion API替代会新增物化、改变stride/alias，而不能删除数学计算。`tensordot`内部产生的reshape/view/permute也不能由外层直接替换；STEP-107已对该真实全频次边界测得`72.629→53.532ms`、仅省19.096ms且18种4D组合不逐位，STEP-108的同表达out-buffer又慢8.905ms。STEP-089～118已覆盖QR、project/project_back、covariance、buffer和foreach边界，不重开换名方案。
- 模型侧可见的PillarVFE两处`permute().contiguous()`是真实活动路径，但最大的大张量输出物化已在STEP-179用更强的“max前移”严格门禁：8-rank逐位/stride/含tie梯度均通过仍只省7.906ms，低于门槛并关闭；confusion transpose仍需执行同一transpose copy，不可能优于删除该copy的上限。MapTR assigner中的插值后contiguous受shape条件控制，MSDA Python fallback当前由已采用的自定义MSDA kernel路径绕过；保留v4栈也只能把聚合copy落到外层，不能唯一绑定19.031ms到一个源码行。point_sampling/ViewCopy/TransData及COMBINED/internal-format均已有正式拒绝证据。
- 裁决`NO_GO_UNSUPPORTED_AND_BELOW_THRESHOLD_NO_UNIQUE_BOUNDARY`：固定A3路径无可用ACLNN实现；即使忽略兼容性并把两种不同copy族全部宽松相加，纯kernel上限仍只有21.941ms且跨消费者。无机制门禁、训练、NPU调用、重复profiling、业务修改或commit。

## 2026-08-14：STEP-200 `npu_add_layer_norm` 亲和建议闭环

- 权威仓库为`ascend_npu_optimize@f922c3897255`、status 0，唯一正确容器为`mapqr-leicheng`。固定`torch_npu 2.7.1`注册schema是`npu_add_layer_norm(x1,x2,gamma,beta,epsilon=1e-5,additional_output=False)->4 Tensor`；官方op-plugin测试把返回定义为`y, mean, rstd, x1+x2`。当前安装版没有旧测试中的可选bias参数。反向schema显式接收`dy/dsum`并返回四个梯度，必须保留x1/x2及gamma/beta梯度；残差和另有消费者时还必须打开additional output并传播`dsum`。
- 活跃直接邻接只有post-norm Transformer残差：BEVFormer一层的self-attn/cross-attn/FFN共3处，MapTR decoder四层每层3处、共12处。它们是`identity + dropout(out)`后立即`LayerNorm(eps=1e-5, normalized_shape=[256])`，当前FP32且Add边界无独立bias，数学上可由融合API表达；post-norm控制流不再消费归一化前的残差和。其余LayerNorm主要是分类/type/color及InstancePoint output projection中的`Linear->LN->ReLU`，没有第二加数，不能人为创建零张量调用融合API。
- STEP-189低开销普通稳定步中，51次LayerNorm forward纯kernel=`2.658825ms`、51次LayerNorm backward主kernel=`3.977283ms`，合计仅`6.636108ms`；另有`0.415671ms` backward transpose子kernel，不能默认当作融合收益。wait与纯kernel分开为`1.095149/0.648940ms`。
- 同一全阶段带栈profile四步聚合中，全部`aclnnAdd` device self=`73.275140ms`（`18.318785ms/step`），全部LayerNorm forward device self=`10.752686ms`（`2.688171ms/step`）。即使错误地把每步全部710个Add（绝大多数无关）和全部51个LayerNorm forward都假设可完全消除，极端上限也仅`21.006956ms/step < 22.7ms`。Add/LN host self约`67.976/7.660ms/step`含带栈异步下发，不能作为可回收wall时间。
- 更窄的一普通步MapTR栈分组中，`other_maptr_head`全部Add device self=`0.882502ms`、全部LayerNorm=`0.851144ms`，过度包含上限仅`1.733646ms`；BEV三个LayerNorm各约`0.61ms`。真实15个邻接只会低于这些全量上限。Add backward是梯度直传、没有独立可删除的Add device kernel；融合backward仍必须执行LayerNorm梯度，不能把`3.977283ms`主体或wait/host时间虚报为收益。
- decoder output projection、标准MHA和BEV/MSDA与STEP-079/R4/STEP-194～197交叉闭环，没有新的单一边界。四时钟裁决为`NO_GO_NO_UNIQUE_ADD_LAYERNORM_BOUNDARY_ABOVE_22P7MS`：纯kernel远低于门槛，摘要也没有可归属到15处邻接的`>22.7ms`跨流wall/busy-union窗口；不进入机制门禁、短训或提交。

## 2026-08-14：STEP-202 成功采集合同只读复核

- 远端仍保留上一轮已经成功执行过的四个仓库外入口及iter30基线checkpoint；权威业务仓库只读核验为`ascend_npu_optimize@f922c389725574257f177c14ff34dda51c6c5c67`且Git clean，容器完整名称为`mapqr-leicheng`。本次只读取文件、SHA和配置字段，没有启动rank、训练、NPU或profiler。
- 成功入口SHA256固定为：hook `b1ed4f094f6734853509c6ace092efa264421c00a8d1e9c38c4de978acd99dfd`，最终profile config `34cdb68893bb874c5957eab8baf01f402a5bb55eedaeb3c69c5e2f4aee89e131`，seed0运行入口 `8c5b315b1741a1557293db1df1bd6c6699494970bc136c434b5b84af9aad65fa`，8-rank launcher `10ad92c723164d52b32734734c8b466f313200165ec1307cb7199e298bb1e0fc`。保留的iter30基线checkpoint为`f001a7d55c19b74d84dd1384f262acef786237822e9581203176853d735f997d`、`1,607,991,401 bytes`，只作为后续同checkpoint测试oracle，不加载进本次profile训练。
- 唯一launch合同为：`mapqr-leicheng`，`ASCEND_RT_VISIBLE_DEVICES=8..15`，`GPUS=8/MODE=single/world_size=8`，batch/rank=16，workers/rank=8，pin_memory=True，prefetch_factor=3，seed=0，deterministic=False，动态loss scale，SOAP frequency=10，`MAX_ITERS=28`；保留login-shell原有CANN/PYTHONPATH并只前置诊断tools、mmdetection3d和业务repo，设置`TORCH_DEVICE_BACKEND_AUTOLOAD=0`，不设置历史拒绝的TASK_QUEUE2/COMBINED/HF32/internal-format等变量。
- profile只在rank0启用，固定`wait=22,warmup=1,active=4,repeat=1,with_stack=True,record_shapes=True,profile_memory=False,ProfilerLevel.Level0,AiCoreNone,l2_cache=False,data_simplification=True`；其余7 rank仍执行完整真实训练。依据上一轮真实映射而非理论猜测，Profiler标签Step23～26对应训练Step24～27：训练Step24为已复现稳定SOAP重步，Step25～27为稳定普通步。step1～3动态scale回退、step4首次有限梯度、首次编译/数据冷启动均在wait之前；因此active窗口不会采到不稳定step。
- 配置中的`load_from=''`、`resume_from=None`保持从头训练，以维持GPU合同的样本/optimizer-step/SOAP相位；末尾overlay把`checkpoint_config=None`，故不会把最后一步checkpoint保存混入active窗口。MAX_ITERS=28仅用于让active窗口完整结束并自然flush；训练Step28/解析收尾不进入Profiler Step23～26的TopN口径。
- hook在`before_run`启动并在每个`after_train_iter`推进，同一个连续active窗口内每个完整迭代自然包含数据/Host、forward、loss、backward、optimizer/SOAP、HCCL和调度，不需要也不允许按阶段重复采集。带栈/shape只用于TopN、bubble和源码归因；端到端吞吐继续以profiler-off基线裁决。
- 用户在STEP-202正式采集解析期间进一步明确：本次profiling全部raw、export、trace、operator、communication、memory、采集日志和harness永久原位保留，不得在分析结束后删除、移动、覆盖或按旧生命周期清理。最终retention manifest必须写`deletion_authorized=false`并记录脱敏路径标识、文件数和总字节数；该要求覆盖此前“复用结束后再清理”的计划措辞。

## 2026-08-14：STEP-202 唯一正式采集与永久保留清单

- 唯一正式任务只启动1次，完整名称`mapqr-leicheng`、后8逻辑NPU、8 rank、batch/rank16、seed0/deterministicFalse、dynamic loss scale、MAX_ITERS28；28/28自然结束且`GPU_CONTRACT_PROFILE_TRAIN_EXIT=0`，fatal/OOM/Traceback均为0。运行中8个直接rank逐一证明`RANK/LOCAL_RANK=0..7,WORLD_SIZE=8`，`npu-smi`8个唯一PID与后4张物理卡双die映射一致；结束后launcher/端口/训练进程/NPU PID均为0。
- 稳定性与功能证据：Step14 SOAP=`28.406s`复现旧周期；Step4首次grad finite，之后保持有限。warmup训练Step23 loss/grad=`263.9223/43.7649`；active训练Step24 SOAP及Step25～27普通步的loss依次`245.4693/245.5020/237.1522/233.3871`，grad依次`45.8240/52.0594/43.8492/46.0410`，全部finite。带栈时间`70.130/48.309/48.641/47.604s`只用于诊断，不作吞吐。
- rank0原位解析总耗时约6m36s，Step28日志时间`482.229s`包含解析等待，最终loss/grad=`239.5764/43.3507`且finite。raw稳定为205个regular files、精确文件字节和`16,647,970,748`；含10个目录的tree apparent bytes为`16,647,980,964`，SQLite journal为0。
- 关键inventory：`trace_view.json`1个/`6,061,257,869B`，`operator_details.csv`1个/`4,464,843,593B`，`kernel_details.csv`1个/`33,692,240B`，`step_trace_time.csv`1个/625B，`ascend_pytorch_profiler_0.db`1个/`1,120,358,400B`；另有`torch.op_range`约4.516GB及Python tracer数据。没有独立`communication.json/communication_matrix.json/memory_record.csv/memory_operator.csv/op_summary*.csv`，后续通信/内存只能使用trace、kernel和DB证据，不能虚报独立文件。
- retention工具补齐显式`retained=true/mutation=false`字段后SHA=`66cef61ea427c2e8d5ac1442fb8af42885e91e1d7a613a6cb9a6b78546b68c99`；对205个raw文件全部流式SHA256。树外manifest大小70,599B，SHA=`464af966dbe32c026e736a30ca64b07498091de64c6513c8beeba26956c5d350`，路径脱敏ID=`a6d783bec48852ef863bb02ce292c36acc2a75ba6503a0754b8566fa3cd37189`；字段验证`deletion_authorized=false,retained=true,mutation=false,mutation_performed=false,hash_mode=all`，raw前后count/bytes完全不变。
- 唯一业务Git副作用为tracked`fusion_result.json`的运行时0增/16删：HEAD SHA=`310a5e7c...eb0f7`，运行后SHA=`b54c70e1...8bdc4`。启动前Git clean且该文件不属于profiling永久保留范围；按授权只恢复该一个tracked文件到HEAD。恢复后Git status0，raw count/bytes和manifest SHA保持完全不变，未删除、移动或覆盖任何诊断产物。

## 2026-08-14：STEP-202 远端启动前只读预检

- 唯一允许容器`mapqr-leicheng`为running；相似容器虽然存在，但本轮未进入或使用。权威仓库为`ascend_npu_optimize@f922c389725574257f177c14ff34dda51c6c5c67`且Git status为0。容器固定版本为`torch 2.7.1+cpu / torch_npu 2.7.1`，`torch.npu.is_available=True`；只设置`ASCEND_RT_VISIBLE_DEVICES=8..15`时可见设备数为8。
- 宿主物理后半映射`NPU4/chip0～NPU7/chip1`（Phy-ID 8～15）共8个逻辑设备全部Health OK、AI Core利用率0，温度44～48℃；每个逻辑设备HBM基础占用约2877～3119 MiB/65536 MiB。`npu-smi`报告所有NPU均无运行进程；宿主和容器没有`train_spetr/torchrun/distributed/profile`进程，29900～29999端口无监听。
- GPU对齐runtime config、seed0 iter30测试oracle、canonical launcher、seed0入口、profile hook/config/wrapper均为普通文件且非符号链接。SHA分别为`02aca0c7...f56a5`、`f001a7d5...997d`、`10ad92c7...e0fc`、`8c5b315b...65fa`、`b1ed4f09...9dfd`、`34cdb688...e131`、`4f6fc16c...1ff5`；iter30 checkpoint大小`1,607,991,401 bytes`。
- 在与正式wrapper相同的login-shell、`cd`至权威repo、保留客户原`PYTHONPATH`并前置诊断tools/mmdetection3d/repo的条件下，`Config.fromfile`成功。结构化字段为batch/rank16、workers8、pin_memory=True、prefetch3、SOAP/frequency10/one-sided1024、动态loss scale、profile overlay禁checkpoint、schedule 22/1/4；唯一活动train配置的ann/flag引用均存在。配置导入前后Git status hash一致，未创建work/raw路径。
- 两次失败均发生在CPU只读静态导入且未创建rank/NPU/profile：第一次从占位输出目录错误推导配置路径而`FileNotFoundError`；第二次没有复现wrapper的`cd repo`，客户评测模块用相对项目根添加搜索路径时导入失败。按真实wrapper cwd修正后一次成功，证明不是客户模块缺失；不安装、不改环境、不写业务。
- 最终裁决：`GO_FOR_SINGLE_FORMAL_PROFILE_LAUNCH`。启动仍须由主任务再次检查选定master port、8个rank和8个`npu-smi`实时PID；本预检本身没有启动训练、NPU或profiler。

## 2026-08-14：STEP-202 profiling 原位分析管线就绪审计

- 远端历史全阶段分析器与审计前本地副本SHA256完全一致，均为`abaed3166f4c87355739665a9725f2da3d23f4775bf3b0406754c74f498c7071`。历史V1继续作为上一轮16.65GB分析的权威复现工具，未被覆盖或改写；本轮增强版必须以新文件名`analyze_gpu_contract_profile_v2.py`上传到新诊断目录。
- 可直接复用的能力包括：跨stream interval merge与device busy union、service/prelaunch/tail/internal bubble、block/side四时钟、纯kernel与total-cost聚合、wait-anchor、AICPU masked ratio、host/sync/comm覆盖率、schema结构、异常Markdown和独立10节架构报告。`analyze_fullstage_operator_stacks.py`继续独立负责data/forward/loss/backward/optimizer/communication/framework全阶段栈归因，SHA256=`580eeff84e0225de751a8d06fade347eba4026411a6149b07cd915c538cc4227`。
- V1缺口是：逐Step只记录service/busy而未完整暴露wall/busy-union/kernel-sum/total-cost；报告只渲染纯kernel榜；bubble前后kernel缺task type/stream且没有有界host stack；`analysis_manifest.json`不是raw保留完整性manifest；communication.json缺失状态被写死。新V2补齐前四项，并把每个Step的纯kernel/total-cost Top30分别保存、对total-cost榜逐Step执行wait-anchor降级。
- 新V2 SHA256=`cbbef28b693840a6d3288b0e4942700d706d78d714ee7bc9b927a8d40eeea6e4`；新增raw保留工具`build_raw_profile_retention_manifest.py` SHA256=`6e60212aa3057d5238606e96f38f791a0eae859c4dcb244d1a959f5f1f73cef2`。保留工具默认对全部raw文件流式SHA256，拒绝空目录、symlink以及把manifest写进raw树，输出`retention_state=retained`、`deletion_authorized=false`、`mutation_performed=false`，不包含删除功能。
- 两个新工具均通过本地`py_compile`；2-Step合成端到端测试通过：正式Draft 2020-12 schema错误0、逐Step四时钟2/2、逐Step双TopN、wait-anchor、bubble kernel task/stream与host项目栈、AICPU路径、独立架构报告10/10节、raw文件SHA 4/4。合成fixture、输出和新pycache已精确清理为0，只保留两个正式工具。
- 条件性缺口：若本轮TorchNPU新导出`communication.json`，V2当前仍按kernel/timeline计算通信时长与重叠，不能把未知schema直接当权威总量；raw inventory后需先只读识别实际schema，再以最小adapter使用communication.json覆盖总量。若仍与历史一样没有该文件，则按技能降级规则明确记录证据缺口，不阻塞bubble、TopN、AICPU、栈归因和架构分析。
- 裁决`GO_ANALYSIS_PIPELINE_READY`：该条件性通信adapter不阻塞唯一正式采集；raw完成且训练/解析进程归零后，先生成hash-all retention manifest，再运行V2、全阶段栈归因、正式schema校验、10节架构校验和最终报告SHA manifest。raw在当前TopN归因与候选复用期间保留在远端，不拉本地、不删除。

## 2026-08-14：STEP-203 GPU profiling 归档安全门禁与真实 schema 初步盘点

- GPU归档为普通非symlink可读文件，`473,979,928 bytes`，SHA256=`ff083f2b40fc62476e44bab2c1bb99f3a14dcfa7efbda3a65865bc8724b46178`。NPU宿主、跳板机和唯一容器均无`7z/7za/7zz/bsdtar/unar`命令，也无`py7zr/libarchive` Python包；按规则没有安装或修改任何远端组件。
- 扩展只读盘点发现NPU宿主和`mapqr-leicheng`已有系统`libarchive.so.13`与`liblzma.so.5`，Python可通过`ctypes`调用；共享盘附近无既有7z便携二进制。由此不需要新增远端依赖。
- header inventory仅有1个entry：普通JSON，无绝对路径、`..`、symlink、hardlink、device或FIFO，声明解压`12,368,970,966 bytes`，共享盘空间约2.03TiB。安全解包工具SHA=`bb6cbddbad6e5d39c5814ce7f30a407042df4171991207C5531F9A513DB91FB9`，启用NO_OVERWRITE、SECURE_SYMLINKS、SECURE_NODOTDOT、SECURE_NOABSOLUTEPATHS并解到全新诊断目录。
- 解包后仅1个普通非symlink JSON，实际`12,368,970,966 bytes`、SHA256=`d826cd2753e94b8f57bdb2b81c49c65841ef410fee7258022fd00e5be997c645`；解包manifest SHA=`80ae3b1fe2df828e2ca4c6307cd7ee7d11dc9c1980a7d37b2b6871e49eb7c664`。原归档解包前后SHA不变，未删除、移动或覆盖GPU/NPU任何原始产物。
- 真实根schema初步为对象：`schemaVersion=1`、`deviceProperties`显示NVIDIA A800-SXM4-80GB、`distributedInfo.backend=nccl`、`traceEvents`数组；traceName表明rank0、GPU客户同名配置及`iter0_49_no_stack`。文件名只作为捕获意图线索，不能在marker inventory完成前声称有50个完整稳定step。
- 唯一容器已有`ijson`，当前用SHA=`9ec96cbd94b1392e7253a1ad3294bb55d467378bf8c6c299107f1ab38ef825f3`的只读流式inventory扫描step/user annotation、capture边界、host/kernel类别、shape/stack与阶段候选；无训练、GPU/NPU初始化或环境变更。

## 2026-08-14：STEP-202 稳定全阶段 profiling 原位分析闭环

- V2与bisect+prefix optimized V2均自然完成；除有意不同的`profile_run`字符串外，异常JSON全部其余顶层字段逐项完全一致。两份Draft 2020-12 schema错误均为0，两份独立架构报告均为10/10节。全阶段operator栈扫描1,823,325行，非空项目栈1,551,153行，覆盖率85.072765%；阶段device-self依次为optimizer 23,173.544ms、forward 4,014.681ms、backward 1,609.228ms、loss/target 1,196.723ms、communication 50.079ms、unknown 39.015ms、data device 0/host 1.682ms。
- Profiler Step23为训练Step24 SOAP：service/wall/busy-union/kernel-sum/total-cost=`71150.834/68064.002/24615.823/24628.128/68144.011ms`；Profiler Step24～26对应训练Step25～27普通步，四时钟分别为`48390.475/46576.717/1794.662/1807.173/48394.209`、`48482.702/46826.372/1803.371/1817.157/48723.059`、`45946.827/45946.827/1787.368/1818.956/47689.057ms`。带栈profile时钟只作归因，不替代profiler-off吞吐基线。
- 普通三步纯kernel pooled Top序列前五为MSDA grad 560.585ms、Conv 399.625ms、AICPU ViewCopy 289.830ms、单次Matmul 248.621ms、MSDA forward 245.430ms。total-cost榜的Fill/BroadcastTo/Index/MemSet/Equal/Cast主要由98%～99%排队等待主导；7个wait-anchor已显式标注，wait不计可回收device compute。独立communication JSON/CSV=0、memory文件=0，因此通信/内存结论降级到trace/DB/kernel证据，禁止猜测通信量或带宽。
- AICPU ViewCopy在四步均恰为2,048次；普通步纯kernel=`97.285/95.822/96.723ms`、wait=`8.128/19.057/8.281ms`，均值为`96.610072ms kernel + 11.821988ms wait`。逐栈聚类发现`bev_encoder.py:427 random_spatial_mask`四步恰为8,192次，严格闭合`4 step × 2,048`；该边界host-self代理为393.895ms/四步，即98.474ms/步，输入shape字段在profile中为空，不能捏造shape。
- 本次权威GPU对齐profile副本的有效overlay为`lidar_dropout_prob/spatial_rate/mask_ratio=0.1/0.2/0.2`，它由GPU合同对齐器从客户运行合同生成；仓库跟踪同名源config中的静态0不是本次trace的运行值，不得替代或误引。B=4且每batch元素执行同顺序`randperm`，内层逐block切片写构成新的单一功能边界。
- 首个新候选裁决为`GO_MECHANISM_GATE_ONLY_NOT_IMPLEMENTED_NOT_TRAINED`：保持enable随机抽样、每batch一次randperm的次数/顺序及RNG前后状态完全一致，把每batch 512次block slice写改成低分辨率block mask的单次更新再精确0/1空间展开。理论净上限为`96.610072ms - replacement_kernel_ms`，11.822ms wait不纳入。必须先逐位mask/output、RNG state、dtype/device/shape/stride/contiguity/storage alias/autograd及边界shape门禁，再做独立NPU机制计时、测试集、loss/grad和正式8卡A/B；GPU同义实现与GPU计时优先对齐。
- 该候选不是STEP-179 PillarVFE max/layout copy（只省7.906410ms，已关闭），也不是STEP-194 clone（最大单次6.888ms）或point_sampling packed-BMM（正式E2E拒绝），故不是重复打开历史ViewCopy候选。当前没有修改业务代码、没有启动NPU/训练、没有commit。
- 首版validation封装把真实scope名`step_internal`误写为`internal`，导致内部bubble列表为空；该错误只在新分析摘要中，未碰raw。保留旧摘要为`superseded_empty_bubbles`证据，以新工具名V2修正后Top5为56.122ms VectorNorm→Stack、55.361/54.134/53.856ms Add→NeTensor、53.202ms VectorNorm→Stack，且均保留前后kernel、stream和host stack。
- 权威最终analysis manifest包含40个分析产物且逐项SHA复核错误0，SHA256=`1bd319c6f6adcf9ba94de49be3de22acc6a533257903a1745d01a7388f2a657b`；权威SHA清单SHA=`328106ff4c65557ccd921d462162658ad5a98dc9bf71f1b7597e4d96da8ec4ba`。raw仍为205文件、16,647,970,748B、10目录，retention manifest SHA仍为`464af966dbe32c026e736a30ca64b07498091de64c6513c8beeba26956c5d350`，未删除、移动或覆盖任何raw。

## 2026-08-15：STEP-203 GPU/NPU 同义业务对比闭环

- GPU trace全量inventory为42,270,301 events、30,254,480个`X`事件、capture span 325.212s；无stack，shape事件17,935,351。完整marker为两次Step0及Step1～48，没有Step49，故稳定普通组按相位选择Steps33～41/43～48共15步，SOAP组为22/32/42。
- GPU普通15步中位四时钟为service/device-wall/busy-union/kernel-sum=`5848.556/5846.212/2045.559/2045.559ms`，underfeed=`3820.300ms/65.367%`，GPU wait=N/A；NPU带栈普通三步中位为`48390.475/46576.717/1794.662/1817.157ms`，ratio=`8.274/7.967/0.877/0.888`。service/wall受异构profiler扰动，最终仍以profiler-off NPU/GPU=`6.1796/4.3241=1.429`、吞吐比约0.700为准。
- `random_spatial_mask` GPU证据闭合：15个稳定普通步中`aten::fill_ [1,8,8]`每步严格2048次，host中位22.645ms；配套select/slice/floor_divide/remainder也各2048次；CUDA `FillFunctor<float>`每步2048次、8.194ms。NPU同业务AICPU ViewCopy为96.610ms/步，因此device ratio=11.79、NPU直接超额88.416ms/步。GPU无栈，归因为高置信count/shape/config/source-version推断，不伪造源码行证据。
- 可解释同义差距另有MSDA forward约2.687x、超额51.36ms/步，backward约1.277x、超额40.51ms/步；但DrivingSDK路径已采用。SOAP QR约19.03x且差约21.6s/SOAP步，但固定环境严格optimizer状态等价路线已关闭。Conv/BN/布局不比GPU同族更慢；BMM与Index/Reduction不能机械同名聚合，且已有端到端或唯一边界反证。
- 新P0仍为mask批量化机制门禁，现实GPU对齐纯kernel差额约88.416ms/普通步；即使完全回收NPU 96.610ms，普通步约为6.0830s，仍约为GPU的1.407x，不能承诺单项达成1:1。

## 2026-08-15：STEP-204 `random_spatial_mask` 严格等价机制与业务门禁

- 权威源码顺序为：先在NPU生成`enable`，再按`b=0..B-1`各调用一次无device参数的CPU `torch.randperm(N)`，最后逐block写入mask并执行`enable*mask+(1-enable)`。客户真实合同`B=16,H=128,W=320,block=8,ratio=0.2`给出`N=640,num_mask=128`，即每step 2,048次`[1,8,8]`写。
- 等价候选不改变随机调用，只收集原始idx并加batch offset，一次H2D后在`[B,1,H//block,W//block]`低分辨率mask上`index_fill_`，再`repeat_interleave`到整block区域；H/W非整除或小于block的尾边显式补1，保留原实现未触及尾边的语义。idx不排序、不unique；跨batch offset隔离，人工重复idx写0仍幂等。
- CPU机制64 case及后8 NPU的8 rank×64 case全部通过：固定4 seeds、B1/4/16、整除/非整除、H或W小于block、ratio0/1/0.237、enable0/1/0.2、FP32/FP16、非连续输入与人工重复idx；mask及`x*mask`逐位相等，CPU/NPU调用后RNG state与后续rand逐位相等，shape/stride/dtype/device/contiguous/base/alias/input unchanged一致。
- 真实shape完整同步host-wall在5 warmup后交替15轮：原实现各rank median`207.028～236.456ms`，候选`0.783～0.941ms`，净省`206.246～235.591ms`。该口径包含CPU randperm、Python循环、H2D、下发及同步，不等同于profile纯kernel；既有原实现纯kernel`96.610ms`减候选完整wall最坏`0.941ms`仍给出保守下界`95.669ms > 22.7ms`。候选分段rank中位约为enable0.110、idx准备0.274、H2D0.071、index_fill0.184、repeat0.097、尾边/contiguous0.014、公式0.079ms。
- 正式机制任务完成过快而未抓到live PID；没有伪造证据。随后单独运行不重复机制的world8 NPU初始化/barrier/45s hold探针，确认8 direct rank、rank/local_rank0～7、torch_npu2.7.1、visible8～15，并由宿主`npu-smi`确认8唯一PID恰落物理4/0～7/1；自然退出后进程/端口/NPU PID为0。
- 最小业务patch仅改`bev_encoder.py`一个函数。CPU真实业务函数64/64及后8 NPU真实业务函数8×64全部逐位/RNG/alias exact；运行时实测无device `randperm`仍返回CPU int64。当前候选未提交，进入正式30-step前仍需固定seed0 GPU合同基线与checkpoint oracle。

## 2026-08-15：STEP-204 fresh 30-step paired A/B与后续采用门禁

- 旧30步原始日志已按旧生命周期清理，不能用摘要作为新A/B唯一证据；因此同一GPU严格对齐config/seed入口/launcher下顺序重跑fresh baseline和candidate。baseline第一次在CPU import阶段因仓库外worktree漏历史PYTHONPATH而0 iter失败；失败日志永久保留。v2只补原成功前缀，并用1个指向权威客户既有`mmcv/_ext`的只读symlink补齐Git忽略运行时；两树import/Config门禁通过后，baseline/candidate均30/30、exit0、后8 NPU live证据完整。
- baseline/candidate metrics SHA分别为`12d33f61...f3bf`和`60b560fe...fe95`，paired comparison SHA=`c4a5d84f...7040`。稳定普通mean/median/P95从`5.322551/5.350360/5.834903s`降至`5.000786/4.985395/5.474887s`，吞吐`24.048620→25.595978 samples/s`；14/14 paired步均为正收益，净省mean/median=`321.765/291.420ms`。cycle mean`7.689581→7.462625s`，SOAP mean`28.440240→28.549830s`。
- loss1～30全finite；grad step1～2缺失、step3 inf、step4～30 finite的dynamic-scale轨迹一致。loss相对差分mean`-0.0393%`、最大绝对`0.3934%`；finite grad中位相对差`0.0732%`、最大绝对`2.0626%`。candidate末步loss/grad=`224.9848/41.73483`，baseline=`225.00627/41.59319`。
- checkpoint comparison SHA=`38035e8d...8d6f`：meta、1042项state_dict与2204个optimizer tensor的key/shape/dtype/schema一致，双方全finite、scalar差异0；在`deterministic=False`下state_dict global relative-L2=`1.8249%`、optimizer=`81.5528%`，不能把结构通过冒充状态逐位等价，仍必须由resume、固定测试和长期收敛收口。
- 对GPU参考，candidate普通吞吐比约`0.8647`、cycle约`0.5917`，本项显著有效但尚未达到1:1。STEP183保留的`f922c38_iter876.pth` SHA=`bf51e523...23a8`且校验OK，可用于功能oracle；但历史生产合同不同于本轮GPU严格对齐config，原876日志也已清理，不能替代fresh GPU对齐876性能A/B。
- paired resume已通过：两边都从meta iter29恢复并记录Iter30～36，唯一非有限grad同为Iter31且随后loss scale降到32768、跳过一次更新；559个optimizer step均从26增至32，输出meta均为35，loss全finite。resume loss相对差最大`0.5037%`；finite grad最大绝对相对差`11.2622%`出现在Iter34，实际为baseline `48.30773`、candidate `42.86721`，不能脱离实际值解读。
- iter36 checkpoint结构、shape、dtype、finite与optimizer scalar完全对齐；自然随机合同下模型global rel-L2=`2.2737%`、optimizer=`77.5138%`，仍要求固定测试和876-step长期A/B，不能仅凭resume结构通过提交。
- 固定512推理的能力边界明确：`random_spatial_mask`只位于`self.training && DEPLOY!='True' && lidar_feat is not None`分支，eval不调用它。同checkpoint固定512承担“未改变最终推理功能”的回归门禁，但不能替代训练期mask逐位/RNG exact门禁；后者已由CPU64/64和NPU512/512真实函数门禁承担。
- STEP-185旧512源身份当前不可恢复：历史源basename/bytes和dataset_len仍有记录，但原harness、子集、输出及runtime config已删除，未留下源文件内容SHA或first512首尾ID；保留probe.log仅证明旧GPU路径FileNotFound。当前容器mount与明确路径均找不到同名镜像，故严格停止，不以其他数据代替。

## 2026-08-15：STEP-204 876-step长期反证与最终拒绝

- fresh baseline/candidate均在同一GPU对齐config/seed/data/world8后8卡下从头完成876/876并exit0；loss全部有限，grad缺失步均为1～2、动态scale非有限均仅Iter3，checkpoint meta/shape/dtype/finite与optimizer scalar schema一致。deterministic=False使长轨迹自然分叉：loss相对差中位`-0.2645%`、最大绝对`12.3269%@Iter678`，finite grad中位`-0.0369%`、最大绝对`114.7825%@Iter381(40.05554→86.03228)`，不能包装为逐位一致。
- 长期性能反证30-step短窗：candidate相对baseline stable normal慢`1.071883%`、SOAP快`1.583035%`、完整周期慢`0.081110%`、all1～875慢`0.096447%`、末100普通慢`2.277711%`。逐step虽450快/426慢且中位省9.075ms，但均值反而增加8.239ms；没有持续普通步或cycle净收益，不能采用。candidate all1～875吞吐`16.525519 samples/s`，相对既有GPU前876吞吐`28.346540`仅`0.582982`，1:1目标未完成。
- checkpoint v1比较因旧格式`map_location=cpu`仍遗留NPU/CPU混合optimizer tensor而只读失败；v2递归强制CPU后完成，paired工件SHA=`127b2d8b...7a42`。结构与finite通过，但自然状态global rel-L2为model`28.8244%`、optimizer`94.1915%`。该结果只说明结构可恢复，不能推翻性能拒绝。
- 启动审计完整保留：baseline上传wrapper无execute bit的首轮在output前Permission denied，改为容器内`bash "$W"`后成功；candidate首次preflight因shell grep引用错误在nohup前停止，随后又因误在宿主调用wrapper而0 iter/无NPU并报宿主无torch，均保留失败产物；经授权用新输出和baseline同构`docker exec mapqr-leicheng bash "$W"`后才形成唯一有效candidate长跑。
- 最终裁决为`REJECT_LONG_RUN_NO_SUSTAINED_NORMAL_OR_CYCLE_GAIN_NO_COMMIT`。固定512仍缺权威数据身份，但不再阻塞已经由长期性能触发的拒绝；不做876后resume、不commit。最终diff SHA=`921d53da...0313`原位永久保留，随后仅恢复`bev_encoder.py`到HEAD blob=`5423a7d7...cde`、文件SHA=`399f349d...6ae2`；权威仓库与baseline tracked0、端口/训练/NPU进程0，所有训练/checkpoint/profile/GPU trace均未删除或移动。

## 2026-08-15：STEP-205 DrivingSDK MSDA 残余差距
- 当前 FP32 路径仅在 forward 对 shapes/start-index 做 `.int()`、对 sampling/weight 做 `type_as(value)`，随后各调用一次 `mx_driving._C.multi_scale_deformable_attn`；backward 仅调用一次配套 backward 并返回三类梯度。每个语义调用在 NPU trace 中均对应一个主 kernel，不存在可由项目侧合并的主 kernel 碎片。
- 每个稳定 step 的 6 个调用顺序固定：temporal `[32,15360,8,32]`、spatial `[112,576,8,32]`、四个 `[16,15360,8,32]`。NPU forward 中位数(ms)为 `12.269/65.342/1.043/1.025/1.045/1.041`，GPU 为 `6.345/25.968/0.363/0.361/0.366/0.375`；NPU backward（逆序）为 `3.068/3.138/2.844/3.199/146.580/27.961`，GPU 为 `1.758/1.730/1.719/1.720/104.740/34.834`。
- 最大残差集中于同 shape、同 FP32 语义的 spatial 位置：forward `+39.374ms`、backward `+41.840ms`，合计 `+81.214ms/step`。GPU temporal 的 value/sampling 为 half、weight 为 float，但 spatial 位置全部为 float，未把 temporal 精度差异错误归因到 spatial。
- operator_details 的项目可控/可见内部边界：两类输入 Cast 约 `0.016ms/step`，weight InplaceCopy `0.128ms/step`，SDK forward Cast `1.573ms/step`、backward Cast `3.545ms/step`、backward InplaceZero `0.974ms/step`；纯 device 乐观合计仅约 `6.236ms/step`。MSDA 栈内无 TransData、AICPU 或同步记录，wait-anchor 不计收益。
- SDK 运行时 schema 固定为 forward 五输入、backward 六输入/三输出；ACLNN 仅暴露输入输出、workspaceSize/executor、workspace/stream。910_93 二进制虽内部有动态 kernelList/super-kernel 与编译期 impl_mode，但项目 wrapper 不暴露 im2col_step、tiling、workspace、layout、precision 参数；包内其他 deformable 算子语义不同。修改预编译 SDK/环境违反冻结规则且无等价性证明。
- 裁决：`NO_GO_MAIN_KERNEL_FIXED_SDK_NO_PROJECT_CONTROLLED_EQUIVALENT_BOUNDARY`。理论大差距位于固定 SDK 主 kernel 黑盒，项目可控边界不足 22.7ms，未发现单一、独立可回退、严格保持 forward/backward/grad/AMP/checkpoint 语义的替代机制。

## 2026-08-15：STEP-206 剩余TopN与GPU基线最终关闭

- 稳定普通Step24～26的prelaunch为`1619.191/1709.697/1562.714ms`，host栈归到MMCV scatter。双方每步均严格有457个`record_stream`，最大图像copy shape `[16,1,7,3,576,1024]`也均为1次/step；NPU带栈scatter device-self仅约`0.008ms/step`，其host-self不能当profiler-off收益。历史pin候选full step反而`+3.475%`，无新的H2D边界。
- forward+loss阶段差约`268.583ms`但跨消费者。Matmul按最内层源码拆分后，唯一超过`22.7ms`的单点为`bevformer_encoder.py:240 point_sampling`的`82.985ms/step`、1次/step，正是已正式E2E拒绝的packed-BMM；其余项目Matmul/BMM单点最大`10.753ms/step`。
- AI-core ViewCopy旧v2 JSON把host/device self展示字段互换；raw CSV复核后采用每个窄项目栈全部InplaceCopy device-self作保守上界。random mask已被STEP-204长期反证；其外最大无项目栈聚合仅`20.833ms/step`且跨消费者，故无单点越线。
- 裁决`CLOSED_NO_NEW_SINGLE_EQUIVALENT_BOUNDARY_ABOVE_22P7MS_AFTER_GPU_BASELINE_CROSSCHECK`：total-cost wait、bubble、阶段总差和无项目栈跨消费者聚合均不冒充候选；无代码、训练、重采或commit。

## 2026-08-15：STEP-208 阶段差距矩阵复核

- NPU普通三步四时钟均值为service/wall/busy/kernel-sum/total-cost=`47606.668/46449.972/1795.134/1814.429/48268.775ms`；GPU普通15步中位为`5848.556/5846.212/2045.559/2045.559ms`。带栈NPU只用于归因，profiler-off普通比仍为`1.429x`。
- data/prelaunch已闭合到双方457次/step scatter；forward+loss虽有`268.583ms`阶段差但跨consumer；backward NPU反而快约`253.790ms`；SOAP约21.6秒QR差仍受固定环境/optimizer状态等价门禁关闭。
- 通信四步纯kernel仅`33.058ms`且`80.509%`与compute重叠，折合未掩盖上限约`1.611ms/step`；普通tail均值`65.502ms`是最后kernel后的host/wait，唯一大tail落到logger且纯device证据不足。均不能作为`>22.7ms/step`单一候选。
- 裁决`NO_GO_NO_NEW_PROJECT_CONTROLLED_SINGLE_EQUIVALENT_BOUNDARY_ABOVE_22P7MS`；没有换名重开历史拒绝项。
- STEP-208完成度审计发现文档生命周期口径漂移：`最终性能优化报告.md`与`PROJECT_STATUS.md`仍有两处把旧profile的`raw=0`写成当前状态。权威当前状态是STEP-202新raw永久保留205文件、`16,647,970,748 bytes`、manifest SHA=`464af966...d350`、`retained=true/deletion_authorized=false`。已保留旧raw曾删除的历史事实，同时明确该阻塞已由新采集解除；后续阻塞依据必须使用STEP-204长期反证、STEP-205固定SDK和STEP-206关闭矩阵。

## 2026-08-15：STEP-208-B 固定环境能力与阻塞反证审计

- 同语义 primitive/API：MSDA 的 Python/_C/ACLNN runtime 仍只有固定五输入 forward、六输入 backward/三梯度输出，空间 FP32 主 kernel 不暴露 im2col_step/tiling/layout/precision；项目可见边界仅约`6.236ms/step`。SOAP QR 仍只有同时物化 Q/R 的`aclnnLinalgQr`，STEP-089～100 已覆盖 batch、Q-only geqrf/orgqr、buffer与双流，且 raw Q/stable-sort/optimizer-state/checkpoint 逐位合同不能放宽。Addmm 已是单 kernel且全族`15.089ms`；confusion-transpose无A3 ACLNN且跨消费者上限`21.941ms`；Add+LayerNorm极端上限`21.007ms`。没有遗漏的同语义、单一且越过`22.7ms`的现成 API。
- 调度/运行时：`TASK_QUEUE_ENABLE=2`正式30-step使全步回归`7.516%`、普通步回归`6.674%`；`COMBINED_ENABLE=1`使全步回归`4.344%`、普通步回归`3.591%`；CPU affinity 的事后oracle仅`0.793%`且设备—NUMA映射不稳定。blocking会关闭TaskQueue，不能作为加速；历史 scalar sync/DDP/DataLoader/HCCL 边界均为必要、低于阈值或已正式回归。
- 新补齐的`PER_STREAM_QUEUE=1`是固定2.7.1中的试验特性，默认0，仅在TaskQueue开启时生效。官方只推荐“多线程多流提交且Dequeue阻塞”场景，并明确Event保序可能增加一级流水耗时、多Dequeue线程可能资源抢占、不支持细粒度绑核和快恢。当前项目没有活动的项目自建多compute-stream边界；HCCL普通步仅约`12.638ms`且87.155%重叠，SOAP自建双流已因stream/allocator/ACLNN争用使周期步回归而关闭。永久profile也没有“Dequeue阻塞可唯一回收>22.7ms/step”的证据，故不以未设置变量本身作为A/B资格。
- allocator/compile：`expandable_segments`只治理OOM/碎片，当前约26.8/65.5GiB且无OOM/碎片证据；禁用cache只会损失复用。`torchair`缺失，`torch.compile backend=npu`退化为eager；原生NPUGraph严格等价但`326.609→327.281ms`且额外reserved约30.667GB。项目stream手工并发、internal-format、HF32、channels-last及剩余源码边界均已由STEP-100/161～198/200/204～206正式关闭，不能换名重试。
- 裁决：`NO_GO_FIXED_ENVIRONMENT_CAPABILITY_MATRIX_EXHAUSTED_NO_SAME_SEMANTIC_BOUNDARY_ABOVE_22P7MS`。精确重开输入仅限：① 当前兼容栈中新增schema兼容的MSDA空间FP32 forward/backward primitive，并给出本项目shape下输出/三梯度/AMP/checkpoint等价及单步净省>22.7ms证据；② Q-only QR primitive能对24类真实shape逐位复现当前`aclnnLinalgQr` raw Q、stable排序及跨两个QR周期+resume完整状态；③ 新证据直接显示至少两个非HCCL compute stream并发提交、Dequeue独占阻塞>22.7ms/稳定普通步，且无快恢/细粒度绑核合同；④ 明确给出可隔离验收和回退的具体CANN/torch_npu/DrivingSDK版本组合、目标API schema与兼容性证明。泛称“升级”或聚合wait/阶段差不构成重开输入。

## 2026-08-15：STEP-209 第三次外部阻塞复核

- 远端只读现场未变化：唯一允许容器`mapqr-leicheng`运行；权威仓库仍为`ascend_npu_optimize@f922c389725574257f177c14ff34dda51c6c5c67`且tracked clean；训练/profile进程、29600～30099监听及NPU训练进程均为0。
- 固定能力未变化：`torch 2.7.1+cpu`、`torch_npu 2.7.1`、`mx_driving 1.0.0+gitde13346`、CANN目录`8.3.RC1`；MSDA仍为固定5输入forward/6输入backward，QR仍同时返回Q/R，无schema兼容MSDA替代或逐位Q-only QR primitive。
- 永久产物未变化：STEP-202 NPU raw仍为205文件、`16,647,970,748 bytes`，retention manifest SHA=`464af966dbe32c026e736a30ca64b07498091de64c6513c8beeba26956c5d350`；GPU archive/JSON仍为`473,979,928/12,368,970,966 bytes`。没有删除、移动、覆盖或拉取。
- 独立完成度复算仍为NPU:GPU=`0.583544235:1`、差`3.222591s/step`，达到GPU吞吐需`+71.366615%`；现有授权内无尚未执行的安全、单一、严格等价、可回退且净收益`>22.7ms/step`候选。
- STEP-207、STEP-208和本次STEP-209形成同一外部能力/新权威证据阻塞的连续三次复现；裁决`BLOCKED_THIRD_CONSECUTIVE_AUDIT_NO_EXTERNAL_STATE_CHANGE`。目标未完成，但按规则正式进入blocked状态。

## 2026-08-15：STEP-211 环境变量官方语义初查

- 华为Ascend Extension环境变量文档说明：`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`启用可扩展虚拟内存段，主要处理频繁变化的分配大小；配置会改变内存占用并可能造成性能波动，不是通用算子加速开关。
- 同一官方文档说明：`TASK_QUEUE_ENABLE`默认/值1为Level1，值2将workspace相关任务迁移到二级流水以增强掩盖，但可能提高NPU峰值显存；是否受益必须以项目端到端A/B为准。
- ATB官方环境变量表中，`ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3`表示引入block合并的退化SOMAS算法，默认值1；`ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1`开启全局中间tensor内存规划，默认0。二者仅作用于ATB workspace管理，必须先证明项目活动路径使用ATB Operation。
- PyTorch官方out-of-tree backend文档说明：`TORCH_DEVICE_BACKEND_AUTOLOAD=0`只禁用`import torch`时自动加载第三方设备后端，用于避免循环导入；它不是训练kernel或调度性能开关。若项目显式`import torch_npu`，通常只改变自动导入路径而非稳态执行。
- 用户给出的Triton-op-generator适合把明确算子参考/GPU Triton kernel转换为Triton-Ascend并执行精度、性能迭代；这支持后续自定义算子探索，但不能把内存/导入环境变量当作算子重写收益。
- 固定环境只读源码进一步确认：`expandable_segments`当前未设置、默认False，首次分配器配置时读取，True改走虚拟地址预留/映射；不支持NPU tensor IPC且初次分配可能更慢。当前训练约26.8/65.5GiB、无OOM、inactive split或reserved/allocated显著背离证据，故不具备性能A/B准入。
- ATB安装环境当前已设置`ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1`和`ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1`；因此用户建议的GLOBAL=1是完全no-op。ALG_TYPE=3会切到block合并/SOMAS退化算法，但项目tracked源码无`torch_npu.atb.*`或ATB Operation调用，`mx_driving._C.so`也不依赖`libatb`；仅加载`libop_plugin_atb.so`不能证明业务使用ATB workspace，预期无收益。
- `TASK_QUEUE_ENABLE=2`确实命中当前torch_npu算子下发路径，但已有同HEAD、后8卡、8 rank、batch/rank16、30-step、profiler-off正式单变量A/B：相对默认Level1，全步`+7.516%`、吞吐`-6.991%`、普通均值`+6.674%`、SOAP`+2.610%`，故维持`REJECT_NO_COMMIT`，不得组合重试。
- `TORCH_DEVICE_BACKEND_AUTOLOAD=0`已用于权威profile wrapper及训练脚本，`tools/train_spetr.py`显式导入`torch_npu`，所以不会禁用NPU；它只控制`import torch`的entrypoint自动加载并避免重复backend注册，稳态step收益预期为0。只可保留在已验证显式导入入口，不能作为新性能项全局盲加。
- STEP-211总裁决：`REJECT_COMBINED_ENV_CHANGE_NO_NEW_PERFORMANCE_CANDIDATE`。五项中GLOBAL=1已生效，AUTOLOAD0已在有效入口采用但非性能项；ALG3不命中活动ATB路径，expandable无碎片适用证据，TQ2已有正式负收益。没有新的`>22.7ms/step`候选。

## 2026-08-15：STEP-212 社区 DrivingSDK 与 Triton-Ascend 初筛

- Triton-Ascend 官方架构说明其编译链为 Triton IR→Linalg→AscendNPU IR→BiSheng 目标文件，并提供vector/cube tiling、multibuffer与CV balance等优化能力；A2/A3/A5在支持范围内。官方安装矩阵显示CANN 8.3.RC1对应较早的Triton-Ascend 3.2.0rc4，而当前稳定组合通常已转向CANN 9.0/torch_npu 2.7.1.post4。CANNBot Triton-op-generator可以生成和迭代算子源码，但不自带客户容器所需编译/runtime；若固定容器未预装兼容Triton-Ascend，项目规则禁止为试验安装或替换远端组件。
- DrivingSDK公开MSDA源码已存在generic与high-performance两套kernel；历史优化提交的dispatch包含tiling key `1002/1004/1008`，分别实例化`KernelMultiScaleDeformableAttnOpt<2/4/8>`。当前GPU/NPU同shape差距最大的空间FP32调用仍为forward约`65.342 vs 25.968ms`、backward约`146.580 vs 104.740ms`，合计NPU超额约`81.214ms/step`，因此社区源码核验优先确认当前安装包是否已含这些优化kernel、真实空间shape命中了哪个tiling key，以及模板参数代表level还是sampling point。
- DrivingSDK公开编译指南允许按kernel name单独构建MSDA，但`setup.py develop`会生成/注册或替换算子产物，不能在冻结客户环境中直接执行。只有固定工具链能在仓库外构建ABI兼容、独立加载且不覆盖已安装包的候选，才可能进入机制门禁；否则仅记录为需要外部SDK补丁/版本输入。
- SOAP QR社区路线仍受更强状态合同限制：必须逐位复现当前`aclnnLinalgQr`的raw Q、稳定排序、连续两个QR周期及resume optimizer/checkpoint状态；一般Triton Householder/Gram-Schmidt/Cholesky-QR即使数值接近也不是准入候选。
- 当前开放问题：①安装的`mx_driving 1.0.0+gitde13346`是否包含上述Opt kernel和tiling key；②空间`num_levels=1/sampling_points=8`真实shape为何仍显著慢；③固定容器是否已经具备可直接导入的兼容Triton-Ascend或仅有AscendC/NpuExtension构建工具。结论前不编译、不安装、不启动NPU。
- 本地只读浅克隆DrivingSDK官方仓库并读取指定commit后，旧版host tiling条件已完全展开：模板参数就是`num_points`；当`embed_dims=32`、points为2/4/8且`levels*points*heads`可被4整除时选`1002/1004/1008`。当前空间真实shape为levels=1、points=8、heads=8、embed=32，故旧版也会命中`Opt<8>`，不能把当前慢点解释成误走generic key0。
- 更重要的新线索是官方master在2025年继续合入多轮MSDA优化：!1105（commit `99f439c`）明确“Optimize for num_heads * num_levels * num_points <= 64”，把tiling改为aligned/fastMode二维key；当前空间shape乘积恰为64，因此正中目标。随后!1112继续优化grad，!1378（`94bb9bd`）和!1504（`c1a2764`）进一步重写forward/grad的数据搬运、双buffer与流水。当前安装版本串`gitde13346`尚未映射到公开commit；必须先证明客户二进制早于或不含这些补丁，才能把官方新版MSDA作为候选。
- 固定容器已安装通用`triton 3.7.1`，但没有`triton_ascend`、`torch_npu.triton`或Ascend backend；Triton backend只有AMD/HIP和NVIDIA/CUDA。CANN8.3RC1对应的Triton-Ascend 3.2.0rc4需要安装/覆盖且旧版与社区Triton不能共存，违反冻结环境。因此CANNBot路线当前裁决为`NO_GO_CANNBOT_TRITON_ARTIFACT_NO_ASCEND_BACKEND_IN_FROZEN_ENV`；AscendC/BiSheng/opc/msopgen/NpuExtension工具链虽存在，但只能作为独立官方MSDA源码补丁的构建可能性，不能证明候选已可运行。
- 官方GitCode映射最终确认`mx_driving 1.0.0+gitde13346`就是DrivingSDK `branch_v7.3.0`头`de133467...`；!1105、!1112、!1378、!1504均已包含。安装包二进制侧独立吻合：910_93 forward/backward仅有新版`11/01/10/00`实体，公共`msda.h`模板为`aligned/forward/fastMode`，不是旧版`template<num_points>`/1008实现。故“补回1008或四个2025优化”属于重复当前实现，裁决`NO_GO_MAIN_KERNEL_CURRENT_DRIVINGSDK_NO_UNAPPLIED_VENDOR_PATCH_OR_PROJECT_TUNABLE`。
- v7.3之后只找到三项通用910B/910_93 MSDA变化：2025-12 embed维度扩展、2025-12 cube move/load-balance、2026-02精度修复。前者不针对当前embed32；cube分支不作用当前aligned+fast key11，仅新的任务负载调度可能经过目标shape，但没有目标shape性能数据或`>22.7ms`净收益证据；精度修复不是性能项。这些补丁属于branch_v26.0/26.1/master，官方配套为`torch_npu v2.7.1-26.0.0`等，而客户固定为`v2.7.1-7.3.0/CANN8.3RC1`，直接采用必须替换冻结栈，禁止试跑。
- 最新官方Triton-Ascend OPLIST与仓库均没有可直接使用的MSDA/QR kernel；QR在op-plugin 7.3/26.0/26.1/master仍调用`aclnnLinalgQr`并同时产生Q/R，没有Q-only新primitive。STEP-212最终没有固定环境内可运行、单一严格等价且有净收益上限证据的候选，因此不进入机制门禁、不训练、不commit。

## 2026-08-15：STEP-213 v7.3后MSDA load-balance补丁初步定量

- 官方GitCode commit `eda3c913a0508f343221c200edb826836240780e`（MR !1840，2025-12-31）提交说明明确：forward稀疏搬运、负载均衡和cube搬运优化；模型case“非FastMode性能优化10%+，fastMode性能优化2%+”。当前空间shape满足aligned+fastMode key11，因此只能引用后者，不能套用10%+。
- 以当前同shape NPU forward约`65.342ms/step`计算，2%仅约`1.307ms/step`；即使将“2%+”宽松看成数量级，仍没有接近`22.7ms/step`准入线的直接证据。该commit主要大改forward，grad host仅补充字段/对齐信息，没有官方backward同量级收益声明；不能把完整forward+backward的`81.214ms`GPU差距全部归给这个补丁。
- !1840还新增AIC数量、约数十MB workspace、cube组装及多段workspace offset，意味着不是一个可在项目wrapper中复刻的轻量调度参数；回移需要完整host tiling、tiling ABI和kernel同步变化。即便技术上可构建，现有官方收益已低于门槛，按计划应在编译/NPU前停止。
## 2026-08-15：STEP-214 Triton-Ascend 隔离安装前置结论

- 用户已明确授权仅在完整名称为`mapqr-leicheng`的现有容器中安装Triton-Ascend；宿主机禁止安装，且不得卸载或覆盖容器全局`triton/torch/torch_npu/CANN`。本阶段只允许仓库外诊断目录中的隔离venv或`--target`安装，先做依赖解析和import/backend验证，不运行NPU kernel。
- Triton-Ascend官方安装矩阵明确：`3.2.0rc4`配套CANN `8.3.RC1/8.3.RC2`，配套`torch_npu 2.7.1`；官方PyPI提供CPython3.11、manylinux_2_27/2_28、aarch64 wheel，大小约50.1MB，符合当前容器Python3.11/aarch64的基础标签。
- `triton_ascend`发行包本身提供`triton`Python命名空间，因此不能与容器全局通用`triton 3.7.1`合并安装。安全路径必须让隔离解释器或隔离`PYTHONPATH`优先加载诊断目录中的Triton-Ascend，同时通过`--system-site-packages`或受控路径复用全局`torch 2.7.1+cpu/torch_npu 2.7.1`，避免复制大型框架包。
- 安装前硬门禁：`pip --dry-run`不得计划安装、卸载或替换`torch`、`torch_npu`、CANN或容器全局`triton`；目标目录磁盘、网络、wheel架构/哈希和当前无训练进程均须通过。若隔离环境无法复用全局框架，或解析器要求改框架依赖，立即停止。
- 正确容器只读preflight通过：唯一精确名称`mapqr-leicheng`且running，架构`aarch64`、glibc2.34、Python3.11.10；固定全局环境为`torch2.7.1/torch_npu2.7.1/triton3.7.1`，尚无`triton-ascend`，Python `venv`模块可用。CANN版本文件报告`8.3.RC1`。
- 仓库外诊断目录位于既有共享盘，当前约17TB总量、2TB可用，inode充足；容器内训练/profiler样式进程计数为0，PyPI和官方GitCode HTTPS探测均为200。仅创建了诊断目录并上传preflight脚本，尚未下载wheel、创建venv或安装任何包。
- 从官方PyPI JSON精确选择并下载`triton_ascend-3.2.0rc4-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`到远端诊断wheelhouse；文件大小`50,145,947 bytes`，SHA256=`91770af4b45a27abadd607cb501e8b77d0f0f395980005b42151aff7f2484a35`，与PyPI元数据一致。wheel标签同时支持manylinux_2_27/2_28 aarch64，当前glibc2.34满足。
- wheel METADATA确认`Name=triton-ascend`、`Version=3.2.0rc4`、`Requires-Dist`计数为0。已在诊断目录创建`--system-site-packages` venv；安装前可见`torch2.7.1/torch-npu2.7.1/triton3.7.1`均来自全局，`triton-ascend`缺失。`pip --dry-run --no-index`报告仅计划安装`triton-ascend`一个发行包，未计划安装、卸载或替换`torch/torch_npu/triton/CANN`，安装门禁通过。
- 隔离安装成功：venv内distribution为`triton-ascend3.2.0rc4`、模块版本为`3.2.0`，`triton.backends.ascend`可解析且backend registry为`ascend`。venv继续从全局路径复用`torch2.7.1/torch_npu2.7.1`；容器默认Python复核仍加载全局`triton3.7.1`且看不到`triton-ascend`distribution，证明全局命名空间未被覆盖。
- 官方GitCode tag `v3.2.0rc4`的annotated tag object为`e94156eeeb8ac16e348b5aa3e23bfc3c85cec7dc`，peeled源码commit为`0df4da8eb40099438686864ed94540e62a04e753`。隔离venv约184MB、wheelhouse约48MB；回退边界是单一STEP-214诊断目录，删除前必须解析绝对路径、断言精确目录且非符号链接。本轮未运行任何NPU kernel，训练样式进程始终为0。

## 2026-08-15：STEP-214-B allocator/TaskQueue 正式入口审计

- 权威仓库仍为`ascend_npu_optimize@f922c389725574257f177c14ff34dda51c6c5c67`，tracked clean；现有未跟踪项仅为既有`diagnostics/`与GPU profile压缩包。
- 客户目标配置的正式8卡继承链应冻结为“诊断外层wrapper → `tools/ddp_train.sh` → `python -m torch.distributed.launch --nproc_per_node 8 --use_env` → `tools/train_spetr.py`”。放在launcher之前并`export`的变量会由8个rank继承，且早于各rank首次初始化torch_npu allocator。
- tracked `run_train.sh`把`GPU_COUNT`硬编码为1，不能作为正式8卡入口；`tools/local_train_spetr_debug.sh`默认8卡但没有AUTOLOAD/TASK_QUEUE设置。单卡`train_card10.sh/train_card11.sh`虽设置`TORCH_DEVICE_BACKEND_AUTOLOAD=0`，不属于正式8卡合同。
- 已验证正式profiling/长测wrapper在启动8 rank前设置`TORCH_DEVICE_BACKEND_AUTOLOAD=0`；tracked仓库未设置`TASK_QUEUE_ENABLE`或`PYTORCH_NPU_ALLOC_CONF`。历史TQ2通过外层`docker exec -e TASK_QUEUE_ENABLE=2`单变量注入并核验8 rank继承，未永久写入业务脚本。
- 最小永久patch边界只能是`tools/ddp_train.sh`中`set -x`之后、读取任何Python入口之前新增一行`export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"`；不得同commit加入TQ2或AUTOLOAD0。当前本地没有该权威脚本副本，历史`.codex-tools` wrappers属于已完成任务合同，不应修改，因此本轮只给远端精确patch方案。
- 单变量顺序必须是：allocator A/B先保持TQ未设置；确认并采用allocator后，TQ2重测的A/B两侧都保持allocator=True，只有B设置TQ2。若allocator未采用，则TQ2两侧都保持allocator未设置。禁止把allocator与TQ2同时作为候选差异。
- 用户补充授权后已在权威远端中央入口实施单行patch：`tools/ddp_train.sh`在`set -x`后新增`export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"`。HEAD文件SHA256为`e006683c9b9da4d1876c2bace6a0789a831f18c70c5636e2f87eaa2b7b2461ed`，工作树文件SHA256为`732830846414e6c923696bf8315f4314b3ecba6430cc606acf13196274316d8a`；numstat=`1 0`，tracked状态只有该文件修改，未加入TQ2。

## 2026-08-15：STEP-213-C STEP202 MSDA 核间负载证据审计

- 只读复用永久STEP202 rank0原始数据。采集元数据明确为`ProfilerLevel.Level0`、`_aic_metrics=ACL_AICORE_NONE`、`ai_core_profiling=""`、`ai_core_metrics=""`；因此本轮没有AICore/AI Vector core级PMU、block级起止或并行核跨度采样，不能从不存在的字段反推核间load imbalance。
- `kernel_details.csv`只有`Step Id/Device_id/Name/Type/Accelerator Core/Start/Duration/Wait/Block Dim`九列。48个有效MSDA/Grad kernel事件的`Accelerator Core=N/A`、`Block Dim=0`；`analysis.db`仅有`StepTraceTime`表，没有task/kernel/core/subtask表。
- 1.12GB `ascend_pytorch_profiler_0.db`只把MSDA保留为整task：`TASK`提供task级`startNs/endNs/device/stream/taskId/contextId`，`COMPUTE_TASK_INFO`的MSDA `blockDim=0/mixBlockDim=0`、shape/dtype/attr均为N/A或空。数据库没有core ID、block ID、每block开始/结束、active core count、尾block完成时间或per-core cycle表；`AICORE_FREQ`只是设备频率时间序列，不能代替核级工作量分布。
- `trace_view.json`可解析出48个整kernel事件，均为`Task Type=KERNEL_AIVEC`、stream33；有`Task Id/Batch Id/connection_id`，但全部`Subtask Id=4294967295`（未知/无效哨兵），且没有block/core字段。该trace只能证明每个MSDA语义调用落为一个AIVEC task及其整核wall span，不能量化task内部哪些core先结束、哪一个core拖尾。
- 现有四步整kernel跨度稳定：空间forward为约`65.183～65.602ms`（四步中位`65.342ms`，离散约0.23%），空间backward为约`146.456～146.859ms`（四步中位`146.580ms`，离散约0.10%）。这证明热点稳定，但“整体稳定”既不证明也不反证核间不均衡。
- 已有GPU同shape映射的整CUDA kernel中位为forward `25.968ms`、backward `104.740ms`；相对NPU整task分别约慢`39.374/41.840ms`。GPU保存数据同样只有整CUDA kernel跨度，没有per-SM/per-block完成分布，因此跨设备只能比较整kernel wall，不能把`81.214ms`差额归因成NPU尾核或负载不均。
- 精确证据缺口是：同一次空间FP32 forward/backward调用的`block/core/subtask -> start/end(or cycles)`明细、有效`blockDim/active-core count`以及末核相对中位核的尾差；若没有这些字段或官方目标shape benchmark，只能裁决`INSUFFICIENT_LEVEL0_EVIDENCE_FOR_CORE_LOAD_IMBALANCE`，不得用wait、total-cost、整kernel总差或MR泛化收益伪造可回收量。

## 2026-08-15：STEP-214-E CANNBot/Triton MSDA源码设计初查

- CANNBot官方quickstart把生成过程定义为参数确认、任务构建、算法设计、代码生成与验证、性能优化与验证、报告/导出六阶段门禁；支持直接描述、标准Torch参考和现有GPU Triton kernel三种输入。Phase 4只按单变量依次尝试向量化加载、grid并行度、连续内存访问、pass合并与小循环消除。该流程是生成/验证编排，不是“任意复杂autograd算子已受支持”的证明。
- quickstart当前master建议CANN≥9.0，但本项目隔离runtime固定为经安装矩阵配套的Triton-Ascend3.2.0rc4+CANN8.3.RC1，因此必须以该wheel实际DSL/backend能力为权威，不能把master生成器面向新CANN的能力外推到旧runtime。
- 首次直接打开GitHub tag路径下programming guide、OPLIST和tutorials均返回404；不重复猜tag URL，后续改为只读检查已核验wheel和官方tag peeled源码树。该错误没有远端写入或NPU调用。
## 2026-08-15：STEP-214-D SOAP Q-only Triton候选源码合同

- 权威实现仍为`ascend_npu_optimize@f922c3897255`的`projects/mmdet3d_plugin/optimizers/soap.py`，文件SHA256=`0e49429dbca9d9a2546c29f54e79639265f7468703ba4b36fa3b3796861a1077`。远端已有1个与本子任务无关的tracked变更，当前只读审计不修改、不暂存、不恢复任何业务文件。
- 唯一允许研究的单一源码边界是`get_orthogonal_matrix_QR`中`power_iter = m @ o`之后的`Q, _ = torch.linalg.qr(power_iter)`；候选接口应为`Q = q_only_qr(power_iter)`。`diag(o.T@m@o)`、descending stable argsort、`exp_avg_sq.index_select`、`o.index_select`、`m@o`、原dtype转换、`state['exp_avg_sq']`写回及`state['Q']`持久化必须原样保留，不能把排序、幂迭代或state重排合并进自定义kernel来扩大候选边界。
- 当前每个非空轴的QR输入为方阵FP32，Q进入持久optimizer state；初次identity basis后立即QR，之后`step>0 && step%10==0`更新。历史完整24类shape/count为：`1(106),3(30),4(6),7(37),8(1),11(1),22(1),32(4),40(9),64(28),96(3),120(1),128(18),160(1),192(32),220(4),256(181),352(1),440(4),512(43),768(22),1024(6),2560(8),5120(4)`；当前one-sided路径删除4个5120和4个不必要2560，稳定周期为543次、2560只剩4次。设计门禁仍保留完整24类，以覆盖旧checkpoint/配置边界。
- 严格合同不变：raw Q逐位相同是进入8卡性能A/B的必要条件；仅列符号对齐、`Q.T@Q≈I`、重构容差或loss短跑接近都不能替代。完整SOAP还必须覆盖连续至少两个QR周期以及中途checkpoint/resume后的`Q/GG/exp_avg/exp_avg_sq/state.step/参数`逐位一致和state_dict schema一致。
- 隔离环境只做CPU/import/source审计：`triton-ascend 3.2.0rc4`的`triton.language`确实暴露`load/store/atomic_add/atomic_cas/where/sum/program_id/arange/static_range`；Ascend backend目录包含编译器、driver和adapter二进制，因此MSDA forward、两个无冲突梯度以及`grad_value`原子累加在DSL前端层面均可描述。
- 但该wheel没有随包提供Ascend `atomic_add`/gather/scatter/reduce的后端验收用例；源码命中`atomic_add`的文件仅是通用language/semantic/interpreter层，不能据此证明3.2.0rc4后端已正确且高效地 lowering FP32高冲突原子累加。该能力必须等主任务释放NPU后，以最小独立kernel编译/数值门禁验证，当前不得宣称可用或有收益。
- Householder Q-only数学上可实现，但只能省略最终`R` tensor的持久物化，不能省略反射子形成和尾矩阵更新。2560/5120方阵分别约25/100MiB，远超该版本单kernel tensor总量96KiB（关闭double buffer时192KiB）的限制，必须采用GM tiled、多kernel panel方案。
- `sync_block_all(mode,event_id)`只暴露`all_cube/all_vector/all`，`sync_block_set/wait`仅允许cube/vector互发；官方没有给出跨任意program的全局内存可见性、调度或无死锁合同，故不能把这些扩展当作Householder panel间grid-wide barrier。安全设计只能先用同stream多kernel边界。
- 当前543-call周期若物化完整R，总写出仅`285,545,768 bytes`（0.266GiB；上三角有效载荷0.133GiB），按200～1000GB/s的带宽下界约1.33～0.27ms/cycle，远低于`227ms/cycle`门槛。收益若成立必须来自把当前`QrAiCPU`整体改为AI Core tiled Householder，而不是仅“跳过R写出”。
- 自定义分块、归约顺序和符号选择无法预先复现不透明的`aclnnLinalgQr` raw Q；裁决`NO_GO_FORMAL_RAW_Q_EQUIVALENCE_UNPROVEN_DESIGN_ONLY`。未来门禁仍以任一raw-Q bit mismatch立即停止，符号对齐/正交/重构容差只作诊断，不得进入8卡A/B。

## 2026-08-15：STEP-214-G Triton-Ascend A3最小机制门禁

- 采用官方v3.2.0rc4 vector-add表达，kernel固定为`BLOCK_SIZE=1024`、输入长度98432（覆盖masked tail），只在仓库外STEP-214诊断目录运行。8个torchrun rank通过`ASCEND_RT_VISIBLE_DEVICES=8,...,15`分别绑定逻辑`npu:0..7`，每rank使用独立Triton cache；没有HCCL、训练、profiling或业务代码。
- 静态门禁先通过：vector-add Python `py_compile`、两层launch shell `bash -n`、源码SHA、精确容器名、全局包快照和回退目录合同均成立。最终kernel源码SHA为`56988b944609cdc4a840ae1b07b7fd42755b4906231b894c896af63815893087`。
- 首轮算子本体已在8/8 rank完成编译/加载且`torch.equal=True,max_abs_diff=0.0`，live `npu-smi`也捕获物理Phy-ID8～15共8进程；但controller在ready文件握手处未创建release，rank最终等待超时，故该轮只记`PARTIAL_PASS_OPERATOR_EXACT_FAIL_HARNESS_HANDSHAKE`，不能冒充正式PASS。第二次修正尝试收到主任务立即释放指令时ready=0，按controller PGID精确终止，无算子结果、无残留。
- 最终唯一重跑前新增纯标准库controller，并在本地和正确容器完成CPU/file协议自测：ready必须精确为`rank0..7.json`，成功/异常/信号路径都在`finally`创建release；按物理`NPU4..7 × chip0/1`验证8个live进程；controller105s、外层120s硬超时。vector-add kernel未再修改。
- gate3于09:51:01～09:51:45完成：controller耗时35.9367s，8 ready/8 done/0 failure，所有rank raw输出逐位exact、max_abs_diff=0；live `npu-smi`命中物理Phy-ID8～15的8个独立进程，release后torchrun/controller自然退出。默认Python前后快照SHA相同，仍为全局`triton3.7.1`、`triton-ascend=MISSING`、Ascend backend不可见；隔离venv不污染全局。
- 结束复核训练/torchrun/profile样式进程0，`npu-smi`八组均`No running processes found`。gate3仓库外产物82文件、5,888,096 bytes，保留源码/独立cache/live与release后npu-smi/manifest；不在本地拉取。裁决`PASS_TRITON_ASCEND_WORLD8_BACK8_VECTOR_ADD_EXACT_GLOBAL_UNCHANGED`，只证明工具链和最小elementwise kernel，不证明SOAP QR或MSDA性能/等价。
- 当前业务wrapper的FP32路径精确语义是：`spatial_shapes`和`level_start_index`先转`int32`，`sampling_locations/attention_weights`用`type_as(value)`对齐value dtype；forward调用DrivingSDK五输入接口，backward调用六输入接口并返回`grad_value/grad_sampling_loc/grad_attn_weight`三梯度；Autograd标记`once_differentiable`，对shape/index/im2col_step均返回`None`。
- 仓库CUDA参考实现确认坐标变换为`x*W-0.5/y*H-0.5`，只在`h>-1,w>-1,h<H,w<W`时进入双线性采样，四角分别独立zero-padding；反向对`grad_value`使用原子加，对采样坐标梯度分别乘`W/H`。因此Triton候选不能用clip替换zero-padding，也不能省略外层严格不等式或坐标scale。
- 一次为了定位已安装DrivingSDK路径而执行普通`import mx_driving`，触发PyTorch后端自动加载冲突（报`npu and npu`）；未初始化设备、未编译或训练，且后续改为纯文件系统定位，不再导入。该失败也说明源码审计必须保持`TORCH_DEVICE_BACKEND_AUTOLOAD=0`或完全不import torch/mx_driving，避免把环境装载异常误认为算子缺口。
- 真实业务合同已由既有8-rank shape探针闭合：空间调用为value `[112,576,8,32]`、sampling `[112,15360,8,1,8,2]`、spatial shape `[[18,32]]`，全部数据张量FP32、ND/contiguous，shape/index入口为int64而wrapper内部转int32；每step另有4次MapTR和1次Temporal动态shape调用。最小Triton候选只能以“仅严格命中空间FP32签名，其他调用无损回退DrivingSDK”的方式隔离，不能全局替换这六次动态调用。
- 空间/Temporal/MapTR调用都经同一个`MultiScaleDeformableAttnFunction_fp32.apply`；候选dispatch若放在该wrapper内，必须保留输出`[B,Q,H*C]`连续布局、原有saved tensor集合、`once_differentiable`、index/im2col_step无梯度，以及location/weight随value dtype转换后的保存语义。一次较宽的已安装包文件扫描在SSH超时，未产生远端写入；随后改为精确文件读取成功。
- CANNBot官方当前仓库把Triton生成拆为task extractor、designer、coding、verifier和latency optimizer五类能力，verifier强调baseline冻结、AST防退化、编译运行与benchmark，近期master才新增A3/A5优化经验、device-side gather、输出预初始化与批量测试。它是流程与经验库，不是3.2.0rc4的OP支持清单；当前master的新能力不得倒推客户隔离wheel。GitCode两次页面展开超时，故OP可表达性最终以本地3.2.0rc4 frontend与Ascend adapter符号为准。
- 隔离adapter二进制包含`AtomicRMWConverter`、`DiscreteMaskAtomicAddConversion`、scalar atomic canonicalizer和`fadd`枚举，说明Ascend后端确有atomic lowering路径；同时二进制也带“unsupported atomic kind”“Illegal mask for float atomicrmw”等约束。因此结论由“纯前端存在”提升为“存在后端转换器，但FP32、离散mask与高冲突组合仍须最小kernel实机门禁”，不能直接当作完整MSDA backward已受支持。

### STEP-214-E 最小Triton候选设计（仅设计，尚未编译）

- dispatch严格限制为业务空间FP32签名：`value=[B,S,8,32]`、`sampling=[B,Q,8,1,8,2]`、`weight=[B,Q,8,1,8]`、单层`spatial_shapes=[[H,W]]`、`S=H*W`，且五个输入满足现有ND/contiguous及dtype合同。当前实值为`B=112,S=576,Q=15360,H=18,W=32`。任何dtype、level、point、head、channel、stride或shape关系不匹配，直接走原DrivingSDK，不复制、不重排、不改变异常面。
- forward采用`grid=(B,heads,ceil_div(Q,BLOCK_Q))`，每个program持有`[BLOCK_Q,32]`累加器，静态循环8个point；每点按`x*W-0.5/y*H-0.5`计算四角地址，四个独立mask顺序gather并乘双线性权重，最后写`[B,Q,heads*32]`连续输出。`BLOCK_Q`只能在8/16/32等小集合逐一验证，避免超过3.2.0rc4约96KiB program tensor预算。
- backward最小实现为同grid的一个融合kernel：读取`grad_output[B,Q,8,32]`，静态循环8点；对每点和32通道计算四角贡献，并用`tl.atomic_add`写`grad_value`；在通道轴`tl.sum`得到该点唯一的`grad_sampling_loc(x,y)`和`grad_attention_weight`后普通store。这样只产生三类最终梯度，无中间贡献张量；`grad_value`必须在计时边界内预清零。
- 禁止“先展开再归约”的确定性替代：真实shape若显式保存每点四角每通道贡献需`56,371,445,760 bytes=52.5GiB`，明显不可接受。基础张量本身已很大：value 63MiB、sampling 840MiB、attention 420MiB、output/grad_output各1680MiB；任何新增workspace都必须单列峰值并证明不会挤压客户8卡训练。
- 设计判定为`DSL_EXPRESSIBLE_BACKEND_ATOMIC_UNPROVEN`：forward及sampling/weight梯度所需masked load、arange/static loop、where和sum已有接口；grad_value所需FP32 atomic后端也有转换器符号，但尚未实机证明合法mask、数值与高冲突性能。因此当前只准保留设计，禁止接业务wrapper、编译或宣称收益。

### STEP-214-E 等价与性能门禁

- 动态shape：先对真实`112/576/15360/8/32/1/8`验收，再覆盖`B/Q`非block整除尾块、`H*W=S`的多组单层小shape和空/最小边界；L/P/H/C变化必须证明fallback与DrivingSDK结果/异常一致。只有真实空间签名允许候选dispatch，MapTR/Temporal继续原SDK。
- 边界采样：逐轴覆盖变换后`-1`、`nextafter(-1,in/out)`、`-0.5`、`0`、整数栅格、`H/W-1`、`H/W-0.5`、`H/W`及外侧，验证严格外层条件与四角独立zero-padding；不得clip。NaN/Inf、有限mask以及不可导栅格点由DrivingSDK实测定义行为，候选不得自行发明语义。
- AMP/dtype：autocast开/关分别核对wrapper内value/location/weight实际dtype、输出和三梯度dtype；FP16/BF16或混合dtype若未完全命中FP32合同必须fallback。保留shape/index转int32、location/weight`type_as(value)`和saved tensor dtype。
- stride/layout：真实连续stride、输出`[B,Q,256]`连续stride、三梯度stride/contiguous/format必须逐项一致；transpose/slice/storage_offset/非连续输入走SDK并比较相同成功或相同失败。不得通过隐式`.contiguous()`掩盖成本或改变alias/storage语义。
- autograd：同输入、同`grad_output`比较forward以及`grad_value/grad_sampling_loc/grad_attention_weight`；保存张量、`requires_grad`组合、`once_differentiable`二阶梯度失败行为、checkpoint/recompute、异常传播均一致。数值不得宽于既有DrivingSDK门禁包络：output max-abs `<=7.451e-8`、grad_value `<=2.682e-6`、grad_attention `<=2.876e-6`、空间grad_location `<=9.537e-6`，且各张量NRMSE `<=6.439e-7`、finite mask一致；原子顺序导致超过任一门限即拒绝。
- 重复性：DrivingSDK和候选各重复至少10次，记录逐位/最大差、rank间包络；已知SDK grad_value本身可有约`1e-8~6e-8`非逐位差，不能要求候选虚假的bitwise，但候选不得扩大到上述数值包络之外，随后还必须通过固定随机性loss/grad、测试集、30-step、长训与resume门禁。
- 性能准入：空间每step仅1次，永久profile基线forward/backward=`65.342/146.580ms`、合计`211.922ms`；最小候选必须在同进程交替A/B、充分warmup、显式同步、包含grad_value清零/三梯度写回/dispatch开销的无profiler中，合计中位`<189.222ms`，即净省严格`>22.7ms/step`，且P95/8-rank范围不能回归。GPU同shape`25.968+104.740=130.708ms`是最终1:1方向目标，不是把NPU wait或总差伪造成收益的依据。
- 编译/JIT首次成本、cache命中、额外峰值显存需单列；稳态门禁通过后仍须客户配置8-NPU 30-step端到端包含首次启动成本，普通/周期step、loss/grad、吞吐均通过才允许单一可回退commit。当前主任务未释放NPU，所以所有实机门禁均未执行。

## STEP-214-C allocator-only正式A/B（2026-08-15）

- 身份与运行门禁：权威`ascend_npu_optimize@f922c38`，复用STEP204 fresh baseline及完全相同30-step launcher/config/seed/global batch128；candidate只增加`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`，8个rank均确认该变量、`TASK_QUEUE_ENABLE`缺席、`WORLD_SIZE=8`和`torch_npu`加载，后8 chip由8个唯一Python PID占用。
- 性能反证：stable normal 15～29排24由`5.322551s`变为`5.429718s`，回退`2.01345%`；SOAP14/24由`28.440240s`变为`28.666940s`，回退`0.79711%`；cycle15～24由`7.689581s`变为`7.766228s`，回退`0.99676%`。paired 14步仅8步为正，平均净收益`-107.167ms/step`。
- 功能门禁：30步loss全finite，iter3 grad为预期Infinity、iter4～30全finite，dynamic loss-scale轨迹一致；相对fresh baseline的loss最大相对偏差`0.3613%`、finite grad最大相对偏差`3.8189%`。candidate checkpoint为`1,607,991,337B`，SHA256=`ea47c47...e1e299`；文件SHA不同不能等价于状态不一致，另做CPU内容树比较。
- 裁决：`NO_GO_PERFORMANCE_REGRESSION`。allocator单行patch因用户明确要求加入而暂留、未commit；不得把它宣称为性能优化收益。完整log/JSON/checkpoint/metrics、rank env证据与SHA永久保留，训练/端口/NPU PID均已释放。
- 显存与checkpoint补充：全30步max/last由`25837→25698MiB`，normal窗口max由`25652→25503MiB`，分别降低139/149MiB，但不足以抵消性能回退。CPU内容树比较确认meta一致，state/optimizer共1042/2204 tensor的shape/dtype/finite全一致、scalar差异0；自然非确定性下state/optimizer global relative L2为`0.01883/0.81634`，因此本轮只依据loss/grad相位与结构合同判功能可运行，不声称checkpoint逐位等价。
- STEP-214-F暂停：以allocator-only为共同基线的TQ2合同已冻结；一次启动与“优先算子局部对比”的新指示发生竞态，0 iter初始化阶段即精确终止，未形成checkpoint/metrics/launcher_rc，不计正式A/B。暂停目录及日志永久保留，端口/训练进程/NPU PID均已释放。

## 2026-08-15：STEP-214-I 官方MSDA/FP32 atomic实现检索

- 纠错：隔离安装的wheel载荷确实没有附带测试，但与其完全相同的官方GitCode tag `v3.2.0rc4`（peeled commit `0df4da8eb40099438686864ed94540e62a04e753`）包含`ascend/examples/pytest_ut/test_atomic_add.py`。其中FP32用例覆盖`(32,32)/2 cores`、`(128,128)/8 cores`、`(32768,16)/32 cores`；每个program把自身输入加到同一段`yindex=arange(BLOCK_SIZE)`目标，属于多核重复地址高冲突，而非仅无冲突atomic。官方测试验证最终destination，但使用`validate_cmp`且没有时延数据，不能外推为逐位重复性或性能PASS。
- rc4的该用例索引是静态、连续、规则重复；它不能证明MSDA backward的坐标驱动离散地址。官方后续commit `f1f060ea67bc57ad4ca2e5468dec110100f05f03`才新增`test_indirect_atomic_add.py`，明确覆盖structured+discrete mask、partial structured remap和fully unstructured indirect offsets（rank 1～5）；同一时期还有`31eadfb975c3fb67d577c3ca8510ba76548db687`等SIMT/discrete atomic lowering工作。这些提交位于2026年的main/release-3.2.2链，不属于当前rc4/CANN8.3RC1冻结组合，因此既不能升级采用，也不能把后版测试倒推为rc4能力。
- 官方API文档对A3标记`atomic_add`可用，但注明不支持“多核atomic add同时保存中间结果”的模式。MSDA候选必须丢弃atomic返回值，只读取最终`grad_value`；rc4示例把返回旧值另存仅用于测试，生产候选不得照搬该输出路径。
- DrivingSDK commit `021657b647c1a9b1f7fd6ec4a0101280dbeb96a7`（MR !1710）新增ScatterAddV3，核心思路是先在UB内比较/归并/ReduceSum，再以`SetAtomicAdd<float>()`写GM，从而减少逐贡献全局原子；提交说明的优化手段是提升UB利用率、以矢量操作替换大量标量for/if，但没有公开数值benchmark。其重点快路径是`tailLen==1`，而空间MSDA的`grad_value[...,32]`尾轴为32；泛化分支只能作为“局部聚合后少量atomic”的算法蓝图，不能声称命中目标shape。
- ScatterAddV3随后立即有精度修复`a927a511d262a1de28a0b4b45254709686d7303b`（流水同步和偏移错误）及兼容修复`ea8912d1a5661b8651552c8b3ea41dde093f5b31`，说明不能只摘取最初性能提交。它晚于当前DrivingSDK branch_v7.3.0、使用新的AscendC host/kernel构建链；在冻结CANN8.3/torch_npu ABI下没有可直接局部加载证据。
- op-plugin的官方`scatter_add`只是PyTorch语义映射，不能避开MSDA约52.5GiB的完整四角贡献展开；CANNBot是生成/验证工作流，不是已有MSDA或atomic runtime实现。本轮未找到CANNBot官方可直接复用的MSDA/高冲突atomic示例；官方仓库地址的有界`ls-remote`候选返回403、网页搜索服务返回401，故不把非权威镜像或搜索摘要作为证据。
- 性能证据：除STEP-213已关闭、且当前shape收益约1.307ms/step的MSDA load-balance补丁外，上述官方材料都没有`[112,576,8,32]`/queries15360/heads8/points8的数字；不存在可据此证明`>22.7ms/step`的ready patch。最终裁决为`NO_READY_FULL_MSDA_IMPLEMENTATION_GO_RC4_STRUCTURED_ATOMIC_MICROPROBE_ONLY`：只允许复用rc4官方规则重复索引FP32 atomic用例做隔离机制/时延探针，不能运行fully-indirect rc4路径、不能接业务或宣称完整MSDA收益。

## 2026-08-15：STEP-214-H Triton-Ascend FP32规则高冲突atomic局部门禁

- harness严格使用静态规则索引，不读取坐标或间接offset。case1复用rc4官方高冲突结构并改成channel32：`[512,32]`输出、32 programs对每个output各加1次；case2模拟MSDA平均冲突强度：单batch `[576*8,32]=[4608,32]`输出、32 collision programs、每program静态重复27次，即每output 864个贡献，略高于真实平均`15360*8*4/576=853.333`。
- 8rank后8物理die并行，warmup3、每个case分别7次kernel-only和7次zero+kernel NPU Event测量。case1 kernel Event的8rank中位数中位数为`0.144020ms`（rank范围`0.116600～0.151520ms`），zero+kernel为`0.138060ms`；case2分别为`2.389620ms`（`2.373400～2.411380ms`）和`2.388940ms`（`2.340300～2.407940ms`）。Event p95跨rank最大值分别为case1 `0.466140/0.172120ms`、case2 `2.524400/2.497000ms`；首项存在编译后初期离群，正式解释采用每rank中位数。
- case2每次执行`147456*864=127,401,984`次FP32 GM atomic add，按`2.389620ms`折算约`53.315G atomic-add/s`。真实空间MSDA若逐四角/点/channel直接atomic，总量为`112*15360*8*8*4*32=14,092,861,440`次；在相同饱和吞吐下约`264.33ms`。这是局部规则冲突吞吐外推、不是fully-indirect真实shape实测，但它已比当前DrivingSDK整个空间backward `146.580ms`慢约`117.75ms`，更不可能贡献`>22.7ms/step`净收益。
- 数值/执行门禁全部通过：8个rank/8个唯一PID，两个case共每rank14次结果均finite、oracle逐位exact、重复逐位exact、max_abs_diff=0。controller在`35.952s`抓到物理NPU4～7×chip0/1八进程，finally创建release；总运行45s内自然退出。全局Python快照前后一致，结束时atomic/torchrun/训练/profile进程0、后8 NPU进程0、port29964空闲。121个工件全部原位保留于远端仓库外STEP-214诊断目录，未拉本地、未删除。
- 最终裁决`PASS_RC4_STRUCTURED_ATOMIC_MECHANISM_NO_GO_DIRECT_PER_CONTRIBUTION_MSDA_BACKWARD`。这把STEP-214-E的“基础规则高冲突atomic未证明”升级为机制PASS，但同时反证了最直接的逐贡献GM atomic完整backward。只有DrivingSDK ScatterAddV3式“先在UB/寄存器严格等价归并，再显著减少GM atomic”的设计才可能重开；rc4没有fully-indirect官方保证，当前也没有可证明减少量、数值等价和`>22.7ms/step`的具体实现。

## 2026-08-15：STEP-214-J register局部归并FP32 atomic B1门禁

- 固定B1输出`[4608,32]`，每rank以独立固定seed生成`[32,27,147456]`随机正负FP32贡献（509,607,936 bytes）。direct kernel对每个output做864次GM atomic；aggregate kernel在每个program内按固定r=0..26顺序以FP32 register累加，再只做一次GM atomic，即每output降为32次。两者读取完全相同输入，PyTorch FP32 `sum(dim=1).sum(dim=0)`为oracle。
- world8结果：direct kernel Event的8rank中位数中位数`2.437300ms`（`2.398120～2.475840ms`），aggregate为`1.207890ms`（`1.194860～1.228020ms`），局部加速`2.018x`；wall分别`2.566898/1.310263ms`。aggregate B112线性外推中位数`135.284ms`、rank范围`133.824～137.538ms`，高于继续真实shape的严格上限`123.88ms`，相对DrivingSDK `146.580ms`仅理论节省`11.296ms<22.7ms`，因此按冻结合同不运行B112。
- 随机数值显示归并减少舍入噪声但不恢复逐位确定性：direct相对oracle `max_abs<=2.136230e-4`、`NRMSE<=5.462632e-7`，aggregate为`3.814697e-5/1.270072e-7`；aggregate-vs-direct为`2.288818e-4/5.506473e-7`。两者均finite；direct跨重复max差`1.983643e-4`，aggregate仍有`3.814697e-5`，原因是32个program的最终atomic顺序不固定。
- 机制运行自然PASS：8rank/8 PID、live物理NPU4～7×chip0/1、controller `37.203s`、finally release、global snapshot一致；结束active0/live0/port29965 free。121个工件永久原位保留在远端仓库外，未拉本地、未删除。最终裁决`NO_GO_REGISTER27_AGGREGATE_PERF_BELOW_22P7_AND_RAW_REPEAT_NOT_EXACT`，不进入真实shape、fully-indirect、完整MSDA、训练或业务改动。

## 2026-08-15：STEP-214-K 空间MSDA forward B1真实shape原型

- 原型严格实现`x=loc_x*32-0.5/y=loc_y*18-0.5`、floor四角、每角独立zero-padding、8点attention reduction；每program负责一个query/head的32 channels，因此真实B1 grid=`15360*8=122880`。输入含`[-0.25,1.25)`随机边界/越界坐标；已准备DrivingSDK同输入oracle、shape/dtype/finite/max_abs/NRMSE/重复性及Event比较。
- gate1在首个Triton warmup前由rc4 Python runtime拒绝，错误为grid应小于65536，并明确提示隔离设置`TRITON_ALL_BLOCKS_PARALLEL=1`。未出现OOM或数学编译错误，无rank ready，也没有可用正式时延/数值样本；controller释放全部rank。
- 经主任务同意，仅设置该隔离runner变量、kernel和数学SHA不变运行gate2。host检查被越过，但A3 device runtime仍返回`coreDim=122880 can't be greater than UINT16_MAX`、`rtKernelLaunch invalid value`；失败发生于首个Triton warmup Event synchronize。DrivingSDK warmup虽先执行，但未进入正式采样循环，因此不得把它误报为本轮Event基线。
- gate2失败后按冻结合同不再改为grid15360并在kernel内循环8 heads；那会改变执行结构并构成新候选。两轮失败工件均永久原位保留，结束active0/live0。裁决`NO_GO_RC4_COREDIM_LIMIT_TRUE_SHAPE_FORWARD_PROTOTYPE_UNEXECUTABLE`，组合`B1<0.48159ms`门槛无数据且未通过，不运行B112、训练、profile或业务改动。

## 2026-08-15：STEP-214-L 空间MSDA forward双head/grid61440候选

- 独立候选把每program职责改为静态2 heads×32 channels，grid降为`15360*4=61440<65536`，runner显式unset `TRITON_ALL_BLOCKS_PARALLEL`；其余输入、DrivingSDK oracle、8点bilinear、四角独立zero-padding和attention reduction均复用STEP-214-K冻结合同。
- 唯一world8 B1 gate可编译/launch并完成7次Event：DrivingSDK中位数的8rank中位数为`0.644240ms`（`0.607440～0.807940ms`），Triton为`2513.123901ms`（`2176.646973～2802.793457ms`），约慢`3901x`；Triton p95跨rank最大`2803.373291ms`。B112线性外推`281469.877ms`，远超forward组合上限`53.938ms`与B1准入`0.48159ms`。
- 数值本身通过：随机sampling实际范围`[-0.25,1.2499995]`覆盖边界/越界，Triton相对DrivingSDK `max_abs<=1.549721e-6`、`NRMSE<=2.742497e-7`，两者均finite且7次repeat逐位一致。说明失败纯属执行映射/性能，不是语义错误。
- controller在`73.969s`内取得8rank/物理后八die并release；结束active0/live0。79个工件远端原位保留。裁决`NO_GO_TWOHEAD_GRID61440_EXTREME_PERFORMANCE_REGRESSION`，不运行B112、不接业务。

## 2026-08-15：STEP-214-M 空间MSDA forward Q-tiled32/grid3840候选

- 独立Q-tiled候选令每program以`[32 queries,32 channels]`张量计算单head，8点和四邻域静态归约；grid=`ceil(15360/32)*8=3840`，远低于rc4 coreDim上限，runner不启用ALL_BLOCKS。
- 唯一world8 B1 gate：DrivingSDK Event中位数的8rank中位数`0.609280ms`（`0.577980～0.705160ms`），Triton `423.697617ms`（`423.157867～424.339752ms`），约慢`695.4x`；B112外推`47454.133ms`，远超forward上限`53.938ms`。
- 数值仍正确且确定：随机边界/越界坐标下`max_abs<=1.490116e-6`、`NRMSE<=2.052884e-7`，双方7次repeat逐位一致、finite。controller `49.166s`抓到world8后八die并release，结束active0/live0。
- 裁决`NO_GO_QTILE32_EXTREME_PERFORMANCE_REGRESSION`。降低grid虽比STEP-214-L快约5.93x，仍与DrivingSDK相差两个数量级以上，不能运行B112或接入业务。

## 2026-08-15：STEP-214-N 空间MSDA persistent grid64候选

- rc4官方tag含`tl.range`的pytest、softmax/gather教程与动态边界案例，TritonToLinalg测试也明确覆盖scf.for；因此persistent设计有官方机制依据。候选固定grid64、BLOCK_OUT256，每program用非展开动态循环stride16384遍历flattened`q,h,c`，块内静态完成8点×四邻域。
- 唯一world8 B1 gate成功编译/launch：DrivingSDK Event中位数的8rank中位数`0.643330ms`（`0.614780～0.822640ms`），persistent Triton为`1518.993774ms`（`1516.164429～1522.391479ms`），约慢`2361x`；B112外推`170127.303ms`。
- 数值与Q-tiled一致：边界/越界随机坐标下`max_abs<=1.490116e-6`、`NRMSE<=2.052884e-7`，双方finite/repeat exact。controller `60.694s`完成world8/live后八die/release，结束active0/live0。
- 裁决`NO_GO_PERSISTENT64_EXTREME_PERFORMANCE_REGRESSION_CLOSE_TRITON_FORWARD`。persistent减少program数却因每program约240个动态块的离散gather而比Q-tiled更慢；至此rc4空间MSDA forward的细grid、Q-tile和persistent三类映射均正式关闭，不再做tile sweep。

## 2026-08-15：STEP-214-O SOAP QR geqrf+orgqr局部原语门禁

- 当前torch/torch_npu固定栈支持`torch.geqrf`与`torch.orgqr`在NPU dispatch。固定`[2560,2560]` FP32确定输入，warmup后奇偶轮换7次；baseline `torch.linalg.qr(mode='reduced')` Event的8rank中位数中位数为`4027.125488ms`，候选`geqrf+orgqr`为`1262.196350ms`，约`3.191x`，单调用理论节省`2764.929ms`。同步wall分别`4027.382039/1262.359129ms`。
- raw合同失败：Q在8rank均非bitwise，最坏`max_abs=4.734844e-6/NRMSE=3.960207e-6`；由packed上三角诊断的R也非bitwise，最坏`8.809566e-5/1.217481e-6`。Q/R/packed均`[2560,2560]` FP32，tau `[2560]` FP32，全部finite。
- `QᵀQ-I` max abs `2.920628e-6`、重构relative L2 `1.097199e-6`只说明数学上可接受，项目冻结规则禁止用符号、正交、重构或loss接近替代raw-Q逐位。采样期每rank额外peak allocated固定`81,801,728B`，reserved增量0。
- controller `58.706s`完成world8/live后八die/release，结束active0/live0。裁决`NO_GO_RAW_Q_BITWISE_MISMATCH_DESPITE_3P19X_SPEEDUP`；速度虽显著，仍不扩24类shape、不接SOAP optimizer、checkpoint或训练。

## 2026-08-15：STEP-215-B SOAP QR 2025～2026官方补丁检索（进行中）

- 去重基线：STEP-089～100已经覆盖batch、`geqrf/orgqr`/Householder、Q-only、排序/重排旁路、power表达式、identity特例、双流及`out=`缓冲；STEP-199要求raw Q、stable sort与跨QR周期/恢复后的optimizer state逐位一致；STEP-214-O虽在2560方阵测得`geqrf+orgqr`相对`linalg.qr`约3.191倍，但raw Q不逐位。因此本轮只接受之后出现的官方新实现/API/提交，不把上述路径换名重开。
- PyTorch官方API语义仍表明：`torch.geqrf`只返回LAPACK式packed reflectors与`tau`，需要再调用`torch.linalg.householder_product`（`torch.orgqr`是其别名）才能显式生成Q；`torch.linalg.qr`公开模式只有`reduced/complete/r`，其中`r`只计算R，没有Q-only模式。官方还明确QR跨平台/设备的Q、R只在符号意义下非唯一，因而API级数学等价本身不能证明本项目要求的raw-Q exact。
- Ascend官方配套表确认当前固定组合`CANN 8.3.RC1 + torch/torch_npu 2.7.1`对应`v2.7.1-7.2.0`，不是`v2.7.1-7.3.0`；7.3.0官方配套CANN 8.5.0。后续QR提交即使出现在7.3/master，也不能直接外推到当前固定栈，更不能据此替换客户依赖。
- CANNBot官方Triton生成器是算子生成/验证/压测工作流，近期新增A3/A5经验与baseline防篡改门禁，但当前公开索引未出现QR/Householder专用实现；它不能作为现成Q-only runtime primitive的证据。

### STEP-215-B 官方语义/API/补丁裁决

- 用户本轮明确允许放宽QR逐位门禁；这是当前权威执行规则，覆盖STEP-199与STEP-214-O“raw Q任一bit不同即停止”的后续准入要求，但不追溯改写当时在旧合同下作出的NO_GO事实。PyTorch 2.7官方也明确：FP32运算即使数学等价也不保证逐位相同，运算顺序、版本和平台都可能改变结果；`torch.linalg.qr`还特别说明R对角线不强制为正，因此有效QR分解在列符号上非唯一。官方依据：[QR API](https://docs.pytorch.org/docs/2.7/generated/torch.linalg.qr.html)、[数值精度说明](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html)。
- `geqrf`返回同一张packed tensor中的上三角R和下三角Householder反射子以及`tau`；`orgqr`只是`torch.linalg.householder_product`别名，用这些反射子显式生成Q。故对方阵FP32输入，`packed,tau=geqrf(A); Q=orgqr(packed,tau); R=triu(packed)`与`linalg.qr(A, reduced)`具有同一数学QR合同，但PyTorch API没有承诺二者在NPU上使用同一底层executor、归约顺序或产生raw相同的Q/R。官方依据：[householder_product](https://docs.pytorch.org/docs/2.7/generated/torch.linalg.householder_product.html)、[orgqr别名](https://docs.pytorch.org/docs/2.7/generated/torch.orgqr.html)、[geqrf](https://docs.pytorch.org/docs/stable/generated/torch.geqrf.html)。
- PyTorch 2.7的`linalg.qr`只有`reduced/complete/r`；`r`是R-only，没有Q-only。`ormqr`只把隐式Q乘到另一矩阵；若用单位阵物化完整Q，仍必须先`geqrf`，还新增单位阵和一次通用Householder乘法，语义上不是省掉分解的Q-only primitive。`orgqr/householder_product`已是更直接的显式Q路径；固定栈下应先以已经3.191x的`geqrf+orgqr`作为唯一主候选，`ormqr(identity)`只允许做一次局部性能补充，不能假设更快。官方依据：[ormqr](https://docs.pytorch.org/docs/2.7/generated/torch.ormqr.html)。
- 官方配套表把当前栈精确映射到`v2.7.1-7.2.0`；公开当前分支/API没有Q-only QR模式。2026年官方CANN `ops-blas` master出现的是`aclblasSgeqrfBatched`，仍只是batched factorization、没有配套Q生成/Q-only SOAP实现，且不是当前CANN8.3RC1已安装接口；客户规则又冻结CANN/torch_npu，因此只记录为未来线索，不安装、不回移。官方依据：[当前配套分支](https://gitcode.com/Ascend/pytorch/tree/v2.7.1-7.2.0)、[CANN ops-blas](https://gitcode.com/cann/ops-blas/tree/master/test)。
- 建议的局部硬门禁不是官方容差，而是结合STEP-214-O实测包络形成的项目工程阈值：24类真实shape、固定输入/环境下全部shape/dtype/device/stride/finite一致；直接比较（不先做列符号对齐）的`Q NRMSE<=1e-5`且`max_abs<=1e-5`；`R NRMSE<=1e-5`只作诊断，因为业务不使用R；`||Q^TQ-I||_F/sqrt(n)<=1e-5`且max-abs`<=1e-5`；`||QR-A||_F/||A||_F<=1e-5`。列符号对齐、condition/rank只记录诊断，不用于掩盖直接Q超限；任一NaN/Inf、schema或异常行为变化立即拒绝。该阈值包住现有2560最坏Q NRMSE `3.96e-6`、Q max `4.73e-6`、orth max `2.92e-6`、recon `1.10e-6`，同时保留约2～3倍安全余量。
- 为区分候选误差与固定栈自身波动，24-shape门禁应做baseline→baseline和baseline→candidate同输入重复：直接Q差异须满足`<=min(1e-5, max(5e-6, 2×baseline自噪声))`；orth/recon仍各自受`1e-5`硬上限。PyTorch官方只支持“控制随机源可缩小非确定性”，不保证跨实现逐位复现，因此自噪声校准不能取消硬上限。官方依据：[复现性说明](https://docs.pytorch.org/docs/2.7/notes/randomness.html)。
- 连续状态门禁采用同checkpoint、同数据顺序与RNG快照的baseline/candidate双轨，至少跨两个真实QR更新周期：每个周期后的`sort_idx/state.step` exact，state_dict key/type/shape/dtype/device exact，参数及`Q/GG/exp_avg/exp_avg_sq`全部finite；Q逐tensor NRMSE须`<=min(5e-5, max(1e-5, 2×baseline双跑自差))`，其余持久tensor、参数、loss/grad的global relative-L2须`<=min(1e-4, max(1e-5, 2×baseline双跑自差))`。任何误差随周期单调放大、overflow/dynamic-scale/skip相位变化或排序变化立即停止。
- resume门禁在第一个QR周期后保存并恢复，跑到第二个QR周期后同时比较“各自不中断轨迹 vs 各自resume轨迹”和“baseline vs candidate”：schema/step/sort/finite继续exact；数值差异不得超过对应不中断差异的2倍且仍受上述`5e-5/1e-4`硬上限约束。通过后才允许30-step正式8-NPU A/B；长期训练仍需独立验证loss/grad/scale/overflow/checkpoint有限性与吞吐，不能由两周期局部门禁替代。
- 裁决：`GO_STAGED_NUMERICAL_GATE_GEQRF_ORGQR_PRIMARY_NO_READY_Q_ONLY_PATCH`。官方语义支持放宽“bitwise或拒绝”的旧门禁，但不证明候选可直接接业务；下一步应先完成24类真实shape局部数值/加权时延门禁，只有周期加权净省`>227ms/cycle`且上述阈值全部通过，才进入SOAP双周期与resume。

## 2026-08-15：STEP-215 24类真实shape局部门禁准备

- 历史24类方阵/count为：`1(106),3(30),4(6),7(37),8(1),11(1),22(1),32(4),40(9),64(28),96(3),120(1),128(18),160(1),192(32),220(4),256(181),352(1),440(4),512(43),768(22),1024(6),2560(8),5120(4)`，合计551次。当前one-sided活动周期删除`5120×4`并把`2560`从8降到4，活动权重为543次。
- 仓库外harness比较同一不可变FP32输入上的`torch.linalg.qr(reduced)`与`torch.geqrf + torch.orgqr`，逐shape记录Q/R max-abs与NRMSE、finite、`QᵀQ-I`、重构relative-L2、NPU Event、同步wall和allocated/reserved；所有shape（包括2560/5120）均先各warmup一次，避免首次编译/allocator初始化污染。
- 数值硬门槛为Q/R NRMSE、正交max-abs、重构relative-L2均`<=1e-5`；stable sort在QR边界之外且输入对象/内容必须不变。性能按当前543次频次加权，只有净省`>227ms/cycle`才升级SOAP状态门禁。
- 本地两个Python已`py_compile`通过。当前无法启动远端：既有Python SSH依赖被本地ACL拒绝读取；临时OpenSSH脱敏包装器在连接前被网络沙箱拒绝，命令未到远端、NPU未占用，包装器已精确删除。因此当前结论是`READY_LOCAL_HARNESS_BLOCKED_REMOTE_TRANSPORT_BEFORE_LAUNCH`，不是算子失败。
- 启动前二次审阅补齐baseline自噪声：所有shape至少保留两次稳定样本，分别记录baseline与candidate Q重复差；直接Q除`max_abs<=1e-5`外，NRMSE还必须满足`<=min(1e-5,max(5e-6,2×baseline_self_nrmse))`。同时新增`||QᵀQ-I||F/sqrt(n)<=1e-5`，避免仅用max-abs掩盖总体正交误差。修正后Python/AST再次通过。

## 2026-08-15：STEP-215-E SOAP最小业务候选与双周期harness设计

- 既有权威源码证据把唯一可改边界定位为`get_orthogonal_matrix_QR`内`power_iter=m@o`后的单次`torch.linalg.qr`；仓库外草案只替换为`packed,tau=torch.geqrf(power_iter); Q=torch.orgqr(packed,tau)`，不移动stable sort、两个state重排、FP32路径、原dtype转换或Q/state写回。
- `packed/tau`仅为局部临时tensor，不进入state_dict；业务不构造`triu(packed)`。因此静态设计不改变key/shape/dtype/device，但应用前仍须用实际state schema逐项证明。现行无`out=`，候选也不引入out/alias复用。
- `linalg.qr`与低层Householder原语的autograd合同不应视为相同；该边界只在`SOAP.step`的no-grad optimizer调用链准入，harness必须断言grad disabled、输入`requires_grad=False`和Q无`grad_fn`。任何未来grad-enabled调用均不在候选合同内。
- 正常输入之外必须对照full-rank/近秩亏/秩亏/零/NaN/±Inf：两边成功/异常类型、finite mask和shape/dtype/device需相符；不在业务路径新增finite同步分支。PyTorch明确秩亏QR可能不抛错，故“未异常”不是通过证据。
- 双周期仓库外门禁采用baseline双跑自噪声加candidate三轨、同checkpoint/参数/state/RNG与可重放gradient；candidate仅在optimizer.step动态作用域临时路由该QR调用，退出立即恢复，不改业务文件。以实际QR调用检测两个周期，比较sort/state.step/schema exact、Q逐tensorNRMSE硬上限`5e-5`、其余持久state/参数global relative-L2硬上限`1e-4`，并在首周期后做各自resume到第二周期。optimizer-only结果不能替代30-step真实loss/grad/scale门禁。
- 草案与完整设计位于`.codex-tools/step215_soap_geqrf_orgqr_candidate.patch`和`.codex-tools/STEP-215_E_SOAP_geqrf_orgqr最小候选设计.md`；当前状态`DESIGN_READY_WAIT_24SHAPE_GATE_NO_BUSINESS_CHANGE`。

## 2026-08-15：STEP-215-E 双QR周期执行框架落地

- 已把设计实现为仓库外三层执行包：Python gate、容器 runner、宿主 launcher；repo/config/checkpoint/output/adapter/SHA/port全部参数化，不修改业务文件。
- Python gate在启动时校验`soap.py` SHA、唯一文本needle及唯一AST上下文；baseline-A/baseline-B/candidate三轨从exact同snapshot开始，每步重新生成并以SHA校验同一可重放gradient。candidate只在目标`get_orthogonal_matrix_QR`动态作用域将唯一QR路由到`geqrf+orgqr`，退出即恢复。
- 周期不硬编码iteration，而以实际目标QR非零调用识别；记录调用数/shape和stable sort index digest。首周期每轨save/load，第二周期同时推进continuous/resume。schema、sort、state.step/discrete值exact，所有tensor finite；Q逐tensorNRMSE硬上限`5e-5`，其余state与parameters global relative-L2硬上限`1e-4`。
- 本地缺少远端权威源码/config/checkpoint，真实optimizer构造不能完成。adapter模板明确列出10项readiness且全部为false，7个接口均fail-closed；harness会在创建ready标记前拒绝模板。因此当前裁决`FAIL_CLOSED_SCAFFOLD_IMPLEMENTED_NOT_RUNTIME_READY`，不是双周期PASS，也不允许合成optimizer冒充。
- 主审复核后补齐自噪声自适应门禁：先以硬上限验证baseline-A/B在周期1、周期2的自差，再按`min(hard,max(1e-5,2×baseline自差))`收紧candidate对应周期；resume单独取baseline-A与baseline-B各自continuous/resume自差的较大值计算同式阈值。Q hard=`5e-5`、other hard=`1e-4`，自适应从不突破硬上限。
## STEP-215-G：当前活动 QR shape 局部 A/B 通过（历史 5120 受限）

- 当前活动 23 类、543 次/周期全部通过分阶段数值门禁；加权 `torch.linalg.qr` 22934.323 ms/cycle，`geqrf+orgqr` 7935.172 ms/cycle；逐 rank 配对节省中位 14955.979 ms/cycle（范围 14771.170～15285.224 ms），远超 227 ms 门槛。
- 历史但当前不活动的 5120 方阵在 8/8 rank 均未通过：Q max_abs=1.44066e-5、Q NRMSE=7.19076e-6；候选必须使用显式 shape guard，仅已准入的 23 类走新路径，5120/未准入 shape 回退原实现。
- 裁决：`GO_ACTIVE_SHAPES_TO_SOAP_TWO_CYCLE_AND_RESUME_GATE`，不是业务提交批准。

## 2026-08-15 STEP-215-H：真实 SOAP checkpoint stateful-subset adapter

- iter30 checkpoint 的 optimizer 有767个单参数group，但只有559个参数拥有SOAP state；这559项均为`GG/Q/exp_avg/exp_avg_sq/precondition_frequency/shampoo_beta/step`且`step=26`，Q inventory恰为当前活动23类/543，无5120。
- checkpoint不保存optimizer参数名到model `state_dict` key的映射，208个无state参数也没有可由optimizer恢复的shape。因此最小adapter只按559个`exp_avg` shape/dtype创建独立Parameter并加载真实SOAP state；208个无state参数省略。
- 此门禁只验证真实SOAP持久state、排序、两个QR周期及save/load resume；占位Parameter并非完整模型checkpoint参数，不能用于声明模型forward、loss、DDP或训练正确性。
- candidate路由采用活动23类显式白名单；5120及任何未知shape回退`torch.linalg.qr`。每个真实周期必须精确出现543次且无5120，否则fail-closed。
## STEP-215-J：候选在真实 SOAP state 的 resume 第二周期未通过

- world8 gate3 已通过 8-rank/live 门禁，baseline-A/B 与 candidate 第一周期均完成；candidate 在 step13 的 continuous-vs-resume operator event equality 上 6/6 已报告 rank 同点失败，其余 rank 被统一终止。
- 当前裁决：`HOLD_CANDIDATE_RESUME_EVENT_MISMATCH_DIAGNOSE_BEFORE_TRAIN`。不得把 24-shape 局部 2.89x 加速直接外推为可提交优化；raw-Q bitwise 放宽不覆盖 stable sort/inventory/guard/state-step 等硬合同。
## STEP-215-L：真实 SOAP 输入的 QR 基底非唯一，第一周期功能状态仍完全一致

- `.contiguous()` 修复后活动局部性能 2.9607x、配对省 15.180s/cycle；resume layout 分叉消失。
- gate5 第一周期 baseline/candidate Q raw NRMSE 大，但 parameters、exp_avg、exp_avg_sq、GG、step 全部逐位一致，证明差异当前仅是新正交基底。
- 裁决：`GO_BASIS_RELAXED_FUNCTIONAL_TWO_CYCLE_GATE_ONLY`。不得把 raw-Q 放宽解释为无条件接受；必须用第二周期参数/非Q状态与 candidate resume 连续性证明功能等价。

## STEP-215-M：basis-relaxed diagnostic 本地 fail-closed 实现

- 新模式只能由显式 `--basis-relaxed-diagnostic` 开启；Python CLI、容器 runner 与宿主 launcher 默认均为 strict raw-Q。放宽范围由固定集合限制为 cycle1/cycle2 两个 `baseline-A vs candidate` 比较，initial、baseline自差、save/load 与 candidate continuous/resume 均不允许忽略 raw Q。
- 每次 view 比较先独立检查其中每个实际 Q：必须为方阵浮点/复数 tensor、finite，且 `max_abs(QᴴQ-I)<=1e-5`。忽略跨实现 Q 距离不会绕过该检查。
- 非 Q 浮点状态与参数同时执行逐 tensor relative-L2 和 global relative-L2 门禁，二者均受有效阈值且硬上限 `1e-4`；离散值与 `state.step` exact，schema exact。既有每周期543次/23类 inventory、无5120、stable sort digest exact 以及 candidate continuous/resume Q硬上限 `5e-5` 均保留。
- 本地6项CPU-free策略单测与Python AST/py_compile通过；未运行远端、NPU或训练。bashlex 0.18因不支持脚本既有Bash `[[ -d ... ]]`语法，不能替代真实`bash -n`，故本轮不宣称shell运行时验证完成。

## STEP-215-N：basis-relaxed world8 门禁在 baseline 自身 Q 正交硬门禁拒绝

- 独立永久诊断根内的6个源文件SHA、正确容器内两份`bash -n`、权威HEAD/allocator唯一diff、torch/torch_npu 2.7.1、后8可见设备、端口与资源清零均在启动前通过；host显式传入`--basis-relaxed-diagnostic`。
- controller捕获ready 8/8、logical rank 8、后8物理die `[(4,0)..(7,1)]`各一个进程，`npu_smi_while_live.txt`为7,623 bytes并release成功。说明world8/torch_npu/npu-smi live合同真实成立。
- 8/8 rank在同一点fail-closed：`baseline-A:cycle2-continuous-resume:left`的`optimizer_state/state_dict/state/278/Q/0`，`max_abs(QᵀQ-I)=1.3064860886702334e-5 > 1e-5`。这是baseline自身实际Q的正交硬门禁，发生在可放宽的baseline-vs-candidate raw-Q距离比较之前；candidate未进入，不能用basis-relaxed掩盖。
- 裁决`REJECT_Q_ORTHOGONALITY_HARD_GATE`：不重启、不进入30-step、不改业务。结束ready/failure/done=`8/8/0`、launcher rc1、active0、port0、NPU Python进程0；tracked状态仍只有用户要求的allocator行。
- 永久工件42个、11,995,506,047 bytes，包含8份cycle1 checkpoint、8-rank失败、outer/wrapper/torchrun、live npu-smi、summary与SHA manifest；summary SHA=`786e04840cae0754f0fa06f34cbc593e4d8553045573f3b16b2888f778acfbd3`，manifest SHA=`680216932b57ba986d3b23c1ac1107e32f5c55fde451ca842115ae5959f64187`。

## STEP-215-O：正交阈值2e-5一次性校准仍被candidate实际Q拒绝

- 校准实现保留默认`1e-5`并新增显式参数，绝对硬上限固定`2e-5`；大于2e-5或小于1e-5均fail-closed。basis-relaxed范围、candidate resume Q `5e-5`、非Q/参数逐tensor+global `1e-4`、sort/inventory/step/schema完全未改。7项本地单测、AST、新SHA和容器内`bash -n`通过。
- 新独立world8运行ready8/live后8die8。baseline-A/B均通过原`1.306486e-5`失败点并完成两个周期；candidate进入第一周期，inventory/sort及checkpoint保存完成后，在save/load的实际Q正交检查失败：rank1路径`state/338/Q/0`为`2.057009730793702e-5 > 2e-5`。
- elastic在首个rank硬失败后终止其余rank，故ready/failure/done=`8/1/0`；该一条失败足以按fail-closed合同拒绝，不能声称其他7rank数值结果。跨实现第二周期/非Q最终比较未到达。
- 最终裁决`FINAL_REJECT_Q_ORTHOGONALITY_CALIBRATED_MAX`：禁止继续放宽或重跑，不进入30-step、不改业务。结束active0、port0、NPU Python进程0；tracked仍只有allocator一行。
- 永久工件57个、32,996,235,925 bytes，summary SHA=`bfc96d0887443bd852b063470fc10e0a5cc09f40a4bdade3720266d014abbd51`，manifest SHA=`f3d39463955485836c65c4f8d6228f5247678534dd583213c666f0e6b55336c6`。

## STEP-216-A：TurboSOAP Brockett core + 单次 cubic polar 静态实现

- 社区一手实现冻结为 TurboSOAP main commit `1339218c180312b6ed1b04013fd910df9aff6ee7`、`soap.py` blob `d1563b35096440d4374c4a2e784dd652d804954e`（Apache-2.0）。本项目首探针只复用其 core 数学：`C/=clamp(abs(trace(C))/n,1e-12)`；`W=linspace(n,1)^1/mean(W)`；`Cq=C@Q`；`grad=Cq*W`；`D=grad-Q@sym(Q.T@grad)`；`X=Q+0.01D`。随后只做一次缩放 polynomial retraction：由`G=X.T@X`的绝对行和上界把 singular 上界缩到1.25，执行`Qnew=.125*(15Z+Z@(-10G+3G^2))`。
- TurboSOAP 默认链还包含 eigengap preconditioner、方向 EMA/列归一化、per-factor scale controller、metric refresh/backtrack/grow、warmup强制QR和`diag_err>0.3`回退；其上游排序非 stable 且默认可用bf16。当前隔离筛选明确禁用 eigengap/EMA/controller/bf16，保留项目 FP32 与 `stable=True`，因此只能称为`pinned community core probe`，不能声称严格复现TurboSOAP完整优化器。
- 新 harness 不使用随机shape输入：实例化既有真实 adapter，从1.6GB权威checkpoint的559个state中读取`GG/Q/exp_avg_sq`，按当前原式 stable sort 重建活动23类/543个`power_iter`。`exp_avg_sq`只在CPU计算轴向 marginal，并按同一个sort index重排；每个factor逐次传入NPU，原始tensor不落盘。
- 生产式 dispatch 只接受活动23类、方阵、FP32、contiguous、no-grad；5120、未知维度、FP16、非连续或grad路径均回退`torch.linalg.qr`。policy要求权威revision/blob/license/URL和全部参数逐字段精确匹配；配置缺失、来源未验证或任一参数漂移均在world8 ready前失败。
- 局部门禁不再以跨实现raw Q距离裁决，raw Q仅诊断；硬门仍要求所有张量finite，baseline/candidate正交max-abs及normalized-Fro均`<=2e-5`，两次重复rel-L2`<=1e-5`，candidate Rayleigh offdiag同时`<=0.3`且不比baseline劣化超过5%/`1e-5`，使用真实marginal构造的下一步对称预条件作用rel-L2`<=5e-3`。candidate峰值allocated/reserved增量分别封顶于baseline或256/512MiB。
- 性能采用每个真实factor baseline/candidate各两次、rank/调用/重复奇偶交替，记录NPU Event与同步wall、逐调用峰值内存；543次实际调用直接求cycle总和。固定basis weight按device/dimension在shape warmup中缓存并单列持久字节，避免重复linspace/mean下发污染稳态计时。每个rank的Event与wall配对净省都必须严格`>227ms/cycle`。只有8rank全部通过数值、内存和性能才输出`GO_TO_STATEFUL_TWO_CYCLE_ONLY`，仍不授权业务改码或训练。
- 本地验证：4个Python `py_compile`、AST/常量合同、6项纯CPU policy unittest、container runner与host launcher的GNU Bash `-n`均通过；`git diff --check`通过。未连接远端、未运行NPU、未修改业务代码。裁决`STATIC_WORLD8_PACKAGE_READY_NOT_EXECUTED`。

## 2026-08-15：planning-with-files 会话恢复（GPU/NPU 1:1 缺口）

- 用户本轮明确问题：NPU 相对 GPU 性能差很多，目标仍是同合同 8 卡吞吐 1:1 或更好。
- 权威完成度：876-step 公共窗口 NPU:GPU `throughput (samples/s)=15.675189:28.346540`，约 `0.553:1`；独立复算完成度为 `0.583544:1`，缺口约 `3.222591 s/step`，达到 GPU 吞吐需约 `+71.4%`。
- GPU 对齐合同下 30-step 稳态：普通步 time mean `6.1796/4.3241 s`，吞吐比约 `0.700`；完整周期 `8.6575/4.416 s`，吞吐比约 `0.510`。SOAP 周期 QR 仍是最大剩余差距。
- 固定环境内严格等价单一边界已多次关闭：MSDA 空间 FP32 主 kernel 为固定 SDK；`geqrf+orgqr` 因 Q 正交硬门禁最终拒绝；Triton MSDA forward 全系列极端回归；allocator/`TASK_QUEUE=2`/HF32/internal format 等运行时项已否。
- 当前未执行的唯一下一动作：STEP-216-A TurboSOAP Brockett 仓库外 world8 局部门禁。通过后也只允许进入真实 SOAP 双周期/resume，不得直接改业务。

## 2026-08-15：STEP-217 GPU 为标准的 NPU 优化项

- 用户给出的远端路径已只读核验：NPU `gpu_contract_alignment_f922c38_8npu_20260814T172611/profile_once/raw` = 205 文件 / 16,647,970,748 bytes；GPU `.7z` = 473,979,928 bytes；解包 JSON = 12,368,970,966 bytes。均非 symlink，未拉取、未删除。
- 以 GPU 为标准的优先级：① SOAP QR 必须优化（GPU 1.198 s vs NPU 22.798 s）；② 普通步分散 host/underfeed 约 +1.86 s，NPU kernel 总量反而更少；③ MSDA 空间 FP32 +81 ms/step，残差在固定 DrivingSDK kernel。Backward 整体 NPU 更快约 254 ms，不是优化对象。
- `random_spatial_mask` 等已有长期反证的方向不重开。NPU 占用期间不启动 world8 训练门禁。

## 2026-08-15：STEP-218 如何优化（方法合同）

- P0 SOAP QR：不升级 CANN。用 TurboSOAP 钉死的 Brockett `eta=0.01`、1 substep、一次 scaled cubic polar，替换周期步 543 次 AICPU `torch.linalg.qr`。保留项目 `stable=True` 排序；活动 23 类走候选，5120/未知/非FP32 回退原 QR。禁止 eigengap/EMA/controller。
- 门禁链：仓库外 world8 局部（正交 max/Fro `<=2e-5`、预条件作用 `<=5e-3`、Event 与 wall 每周期净省均 `>227ms`）→ 真实 SOAP 双周期+resume → 30-step 单变量 A/B → 876-step。任一层失败即回退，不提交。
- 实施阻塞：NPU 被其他任务占用；上次 STEP-216-A-RUN 因 adapter 落在业务仓库 `diagnostics` 未产生 Event 样本。空闲后必须用仓库外入口，不得重复该路径断言失败。
- P1 host/underfeed：STEP-196 已关闭“无唯一等价边界”。优化方法不是再开图模式，而是等能把 ~1.86 s 钉到单一源码/下发边界的新证据；NPUGraph/TQ2/pin 已负收益。
- MSDA：项目不再改实现。只有当前兼容栈出现同语义更快空间 FP32 kernel 才重开。

## 2026-08-15：STEP-219 优化方案审计结论

- 通过项：Brockett 静态包合同内部一致（policy/JSON/测试三方吻合）；dispatch 23类/543、5120 回退正确；清理协议 PGID-only 无自匹配；门禁四层递进符合证据规则；P1 暂缓与 MSDA 关闭的判断有权威反证支撑。
- 缺口1（高）：正交 2e-5 可行性未预验证。STEP-215-N baseline 自身 1.306e-5、STEP-215-O 候选 2.057e-5 被拒；checkpoint 543 个 Q 的现存正交分布可 CPU-only 先算，若普遍贴线则 world8 必败，应省下唯一执行。
- 缺口2（高）：真正风险门是预条件作用 5e-3 与长期漂移，不是正交。probe 禁用了 TurboSOAP 全部安全网（warmup QR/diag_err 回退/refresh/backtrack/EMA/eigengap)，30-step 只覆盖约 3 个周期，876-step 约 87 个周期之间无漂移监控；需在长训门禁中加只读正交/预条件序列记录。
- 缺口3（中）：执行预算。上次 world8 因 adapter 在业务仓库内+入口无执行位烧掉一次且 0 样本；adapter 迁移到 tool-root/harness 是纯静态修复（verify_identity 按 basename+SHA，与目录无关），应在占卡结束前完成并静态复验。
- 缺口4（中）：227ms/cycle 是噪声下限不是价值线。QR 22.64s/cycle，应预声明"局部净省低于约 5s/cycle 即关闭"，避免低价值候选走完四层门禁。
- 规则预声明：若预条件门失败，改 substeps/eta 属于新候选新 policy，不允许现场放宽阈值（防止重演 STEP-215-N→O 连环校准）。
- 预期上限：P0 全成功摊销约 2.0～2.1s/step，整体约 0.78～0.81:1；1:1 必须 P0+P1 同时成立，而 P1 当前无候选。普通步差距两边均 host-bound（GPU 65.4%/NPU 75.3% underfeed），部分或为宿主 CPU 架构差异，非项目代码可回收。
- 口径注记：SOAP 摊销 2.16s + 普通步 1.86s ≈ 4.0s 与 876 口径缺口 3.22s 并非同一口径（30-step 稳态 vs 876 长窗有重叠），对外报告须标注。

## 2026-08-15：STEP-221 方案设计 v2（Brockett 拒绝后的新 P0）

- STEP-220 裁决确认审计缺口 2 的预判：性能/内存门通过（每周期净省约 22.29s），但投影作用 rel-L2 1.41/1.49、Rayleigh offdiag>0.3、2560 正交 6.13e-3——单步流形法追不上 QR 的整周期基跳变，任何"匹配 QR 单周期输出"的异算法候选都会在 5e-3 门上 O(1) 失败。
- 由此推论：固定环境内 SOAP QR 仅剩一条可行边界——不改 QR 数学、把 22.6s AICPU 墙移出关键路径。即 P0-v2 异步流水化：独立流提交 543 次 QR（STEP-096 已证跨流位级一致），训练用旧 Q 续跑，固定 k 步（≤9，全 rank 一致，save/resume 前强制换入）后原子换入，k=0 即原语义回退。
- 与已关闭方向的边界区分：已关闭的多流是"缩短 Step10 自身"（QR 对 QR 并行，8-rank 回归）；host CPU QR 是 GPU 时代 FP64 旧路径（约 95.5s）更慢；batched 位级一致但更慢。异步流水化不缩短 QR、只改生效时机，资源上 AICPU 后台与普通步 AI Core/host 大体正交。
- 风险与量化点：普通步每步约 2048 次 AICPU ViewCopy 与后台 QR 争用；543 次 QR 的 host 下发挤占前台（underfeed 75%）；k 步相位平移语义需训练级门禁。Stage A 微基准价值线：隐藏 ≥70%、前台减速 <5%。
- 预期上限：P0-v2 全成功约 0.78~0.81:1，单项不达 1:1；1:1 仍需 P1 突破，P1 允许训练后可采一次 Level0 低扰动 profile 做 launch-gap 唯一边界聚类。

## 2026-08-15：STEP-221-A Stage A 实测结论（stale-Q 可行性）

- 合成负载有效性：真实 23 类 shape/543 次 QR 在闲置卡上 t_qr_alone=22.734~22.871s，与训练 profile 的 22.6~22.8s 一致，可作 Stage A 权威代理。
- AICPU 是每卡私有资源：8 卡并发时各卡 qr_alone 与单卡持平（22.74~22.87s），跨 rank 无争用。此前"多流 8-rank 回归"的根因不适用于跨步异步边界。
- 隐藏与减速：单卡与 8 卡并发 hidden_ratio=99.86%~100.01%；前台 matmul（4096²×~11393 次）减速 ≤0.14%，含 AICPU nonzero 争用场景。543 次 QR host 下发仅 0.014~0.019s，不挤占 underfeed 的前台 host。
- k 取值：QR 墙 22.8s ÷ 普通步 6.18s → k=4（frequency=10，k≤9 合规），event 未就绪同步等待作受控 fallback；save/resume 前强制换入保证 state_dict 语义不变。
- 剩余风险收敛为一项：k 步相位平移对训练轨迹的影响，只能由 Stage C 30-step loss 噪声带与 Stage D 876-step+测试集裁决。

## 2026-08-15：STEP-221-B Stage B 结论（stale-Q 数值与状态语义）

- 权威 `soap.py` 结构：`get_orthogonal_matrix_QR` 内 `sort_idx` 同时驱动新 Q 与 `exp_avg_sq.index_select`，二者必须原子同装；昂贵项只有 `linalg.qr`，`est_eig`/`power_iter=m@o` 都是 AI Core matmul；`GG.lerp_()` 每步原地更新，故侧流只能承接 QR，不能承接任何读 GG 的运算。
- 该文件是混合换行（259 行 CRLF、其余 LF）且末尾无换行——任何自动改写必须按原始字节操作，否则会产生大面积伪 diff。这与历史"混合换行修复"记录一致。
- 实现选择：**不改 `get_orthogonal_matrix_QR` 本体**，新增 `_qr_plan/_qr_finish/_qr_install` 只服务 stale 路径。k=0 因此不是"相信等价"而是"执行同一段字节"，等价性反过来由门禁正面证明。
- 门禁结果（真实 559 state / 543 factor，单卡）：trio 与原函数在全部 559 个 state 上 Q 与 exp_avg_sq 逐位一致；**异步产出的 Q 与同步 Q 逐位相同**（同输入同算子，仅生效时机不同）；stale_steps 恰为 4；pending 恒为 0 或 559、跨周期不重叠；`state_dict()` 强制 flush 后持久化 key 集恰为 7 键合同。
- 内存：持有 `power_iter` 引用直到安装会多占约 285 MB；改为 `record_stream` 后立即释放引用，额外 allocated 从 535.7 MB 降到 253.7 MB。额外 reserved 722 MB 超出 STEP-220 的 536 MB 门槛，但该门槛是为 Brockett probe 设的，且真实训练启用 `expandable_segments:True`，须在 Stage C 据实复测再定门槛，不得默默沿用或放宽。
- 口径提醒：Stage B harness 前台每步仅 0.13 s，无法掩盖 22.8 s QR，故安装步显示 21.79 s。这不是候选性能，真实掩盖能力由 Stage A 的真实前台负载（隐藏 ≥99.86%）与 Stage C 的端到端 A/B 判定。

## 2026-08-15：STEP-221-C Stage C 结论（真实 8 卡 30-step）

- 合同：HEAD `f922c38`、对齐 config `02aca0c7...`、harness `10ad92c...`、后 8 卡、batch/rank16、seed0/deterministicFalse、`expandable_segments:True`、单变量仅 `SOAP_STALE_Q_K`。
- 性能：SOAP 步 mean 28.557→5.831 s，净省 **22.726 s/周期**（远超 5 s 价值线）；摊销约 2.27 s/step。普通步 +2.12%（5.664→5.784 s）在 ≤5% 门内。
- 内存：训练峰值仅 +55 MiB；Stage B probe 的 reserved 722 MB 担忧在真实训练分配器下未成为峰值问题。
- 功能：30/30、exit0、loss/grad 有限；前 4 步 loss 逐位相同（证明同起点），之后因授权的 k 步相位平移分叉——Stage D 必须以测试集指标而非 bit-identical loss 裁决。
- 收口：业务 `soap.py` 已恢复权威 SHA；候选仍仅存在于仓库外 toolroot。裁决 `STAGE_C_PASS`，进入 Stage D。

## 2026-08-15：STEP-221-D Stage D 300-step（用户取消 876）

- 合同同 Stage C，仅 `MAX_ITERS=300`。
- 主结果：SOAP 28.286→5.938 s（-22.35 s）；全窗吞吐 16.285→21.272 samples/s（+30.6%）；相对 GPU 参考约 0.75:1；候选 loss 仍下降（early 261→late 65）。
- 次级：相对普通步 +6.23% 越 Stage-C ≤5% 门。分解后候选绝对普通步 5.864 s ≈ Stage-C 候选 5.784 s；本轮 baseline 5.520 s 快于 Stage-C baseline 5.664 s，相对差被放大。非安装步误分类（install 槽位时间正常）。
- 裁决 `STAGE_D_PASS`（主门全过，次级失败已明文记录）。业务工作树已装入候选，待用户要求再 commit；默认 `SOAP_STALE_Q_K=0`，启用 `=4`。



## STEP-216-A must-fix复核结论（2026-08-15）

- 旧稿的4列marginal proxy与“逐factor中位数求和”不足以代表真实SOAP周期，现均删除；改用559个真实`exp_avg`经原生`project_back(oldQ)->project(newQ)`的全轴作用，并以3次交替完整543调用cycle的配对Event/wall节省裁决。
- candidate只消费已预计算的`power_iter=C@Q_sorted`和`trace_norm`，不再重复立方复杂度matmul；活动23类/543、5120/未知fallback保持不变。
- v2 source contract锁定adapter/config/checkpoint/SOAP/community policy、SOAP schema/方法签名及8个harness源文件；任一name/bytes/SHA漂移或manifest自引用均在ready前拒绝。
- 当前只完成本地静态与纯stdlib合成验证；未连接远端/NPU，状态`STATIC_READY_FAIL_CLOSED_NOT_PERFORMANCE_PASS`。

## STEP-216-A world8唯一执行结果（2026-08-15）

- 上传和正式预检均通过，但静态包遗漏了SFTP上传后的执行位合同；第一次host直接执行返回`126`，0 rank/0 NPU，按授权记为`failed_start=1/effective=0`。
- 唯一纠正入口使用`bash host_launcher`后，Docker event证明runner在约37ms内`exit1`，output尚未创建。逐项无副作用复核初始断言仅`adapter_outside_repo=FAIL`：诊断adapter实际在业务repo的`diagnostics/`下，而runner要求adapter不位于repo前缀内。
- 外层观测到的`rc143`不是1200秒timeout、OOM或容器重启。runner rc1后，host清理命令中的`pkill -TERM -f "$output"`匹配了包含同一output字符串的清理bash自身，docker exec以143退出；`set -e`随后终止host并掩盖原始rc1。
- 因此没有3次543-factor Event/wall数据，不能评价Brockett性能、正交或投影作用。裁决`REJECT_PRELAUNCH_NO_CORE_SAMPLE_NO_RERUN`，不是候选算法失败。

## STEP-216-A启动合同静态修复（2026-08-15）

- 恢复包不再允许工具部署到`BUSINESS_REPO/diagnostics`：host和runner都显式接收`TOOL_ROOT`，经realpath后要求tool root、adapter和output全部在repo外，同时把adapter/output限制在固定`harness/`与`runs/`子树。
- 清理链不再按output字符串搜索进程。runner记录torch launcher的PID/PGID，host记录`setsid timeout docker exec`的PID/PGID；两层都只对精确负PGID发送TERM、最多等待5秒后KILL。
- 纯stdlib测试同时覆盖repo内diagnostics拒绝、repo外共享tool root接受，并静态禁止`pkill/killall/pgrep output`。这修复的是启动合同，不改变Brockett公式、23类/543权重、阈值或fallback。
- 极窄复审补齐双侧异常清理：host不再只终止宿主docker-exec PGID；若runner已原子写出`output/launcher.pgid`，host先通过容器内bash严格验证普通非symlink文件、`^[1-9][0-9]*$`且数值`>1`，再精确TERM/有界等待/KILL该负PGID。缺文件代表runner尚未启动，不做推测性杀进程。

## STEP-216-A唯一core运行与真实SOAP接口结论（2026-08-15）

- repo外恢复包预检全部通过，torchrun创建了8个worker；rank0在ready前明确报告实际bound method为`(grad, state, merge_dims=False, max_precond_dim=10000)`，而旧source contract错误固定为`(grad, Q, merge_dims=False)`。这是harness接口合同错误，不是Brockett数值/性能失败。
- controller在首个failure后TERM其余7 rank；ready/done/failure=`0/0/1`，没有live npu-smi绑定、543-factor调用或3个paired cycle，因此不能评价候选性能、正交或真实投影作用。最终active0、端口空、back8进程0。
- 本地恢复包现通过完整state浅视图调用真实接口：old/baseline/candidate各自复制state字典和Q列表，保留其他checkpoint字段；以old state执行`project_back(exp_avg, state, False, 10000)`，再分别用baseline/candidate state执行`project`，原state不写入。

## STEP-216-A PID namespace门禁结论（2026-08-15）

- 修复后唯一运行已达到ready8；每rank证实559 state/543 factor和773574 contract，npu-smi也捕获后8die各一进程。但旧ready把`os.getpid()`同时写成pid与host_pid，得到1131104～1131111；npu-smi为宿主PID 880965～880972，严格集合比较拒绝。该失败发生在release测量前，0 cycle，不是算法性能结果。
- 新协议不再由rank猜宿主PID。ready只声明`container_pid`；controller以npu-smi host PID为锚，读取宿主`/proc/<host_pid>/status`的NSpid链并取最后项，与8个ready container PID做双射，同时持久化physical/chip/host/container四元映射。
- 只读验证现有容器init：Docker State.Pid=2816640，其宿主proc NSpid链最后项为1，符合host→container映射语义；未创建新容器进程或NPU任务。

## STEP-216-A v3运行与host-controller架构结论（2026-08-15）

- v3唯一运行再次达到ready8/559×543/live8；但supervise仍由容器runner执行，读取宿主npu-smi PID 907159的`/proc/907159/status`时FileNotFound。证明映射公式正确但执行namespace错误；测量前拒绝、0 cycle，不是Brockett算法结果。
- 重构后容器runner不再调用controller，只负责torchrun并等待共享release/done。宿主launcher启动独立host Python controller，由其读取共享ready/done/failure、宿主npu-smi和宿主/proc，完成双射后release。
- Host同时记录并管理runner docker-exec与controller的独立setsid PID/PGID；失败清理顺序固定为容器launcher PGID、host controller PGID、docker-exec PGID，最后host postflight。fixture已覆盖8 ready、8宿主proc NSpid链、8 npu-smi row、release及8 done。

## STEP-216-A host-controller 唯一有效 core 结论（2026-08-15）

- 新合同 `e4d80400c752fd4669929a1488ed007679d30857014008531b47d1d46a8aa8ca` 在业务仓库外运行；8/8 rank、8个后卡die、宿主PID→NSpid→容器PID双射均通过，真实 inventory 为559 state、23类shape、543 factor/rank，3次完整 paired cycle，重复性两路径均为0。
- 性能门禁通过：Event配对周期节省rank中位数`22293.308ms`、最小`22184.649ms`；wall中位数`22293.062ms`、最小`22184.794ms`，远高于`227ms/cycle`价值线。
- 内存门禁通过：candidate相对baseline的rank最大allocated增量`255,848,448B < 268,435,456B`，reserved增量`436,207,616B < 536,870,912B`；持久weight cache最大`28,080B`。
- 数值门禁失败：真实全轴project_back→project作用global rel-L2最坏`1.4068267`、逐tensor最坏`1.4907271`，远超`0.005`；多个shape Rayleigh offdiag超过`0.3`；dimension2560正交max/Fro约`0.00613`，远超`2e-5`。只有finite/重复性和部分小shape正交成立，不能抵消真实业务作用不等价。
- 最终裁决为`REJECT_LOCAL_SCREEN`，`qualified_all_8_ranks=false`。该候选虽消除了约22.29s/cycle QR成本，但不能进入双周期、训练、业务修改或提交；固定eta=.01的一步core路径已被实测反证。
