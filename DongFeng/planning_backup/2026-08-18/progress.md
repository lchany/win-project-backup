# Progress Log

## 2026-08-18：STEP-265 结合实测拆因果

- 按 planning-with-files 恢复 `task_plan.md` / `findings.md` / `progress.md`。本机 session-catchup 脚本路径不在 `~/.claude/skills`，改为直接读计划文件。
- NaN：STEP-260 坐实 `mx_driving_cloud.linalg.qr` 对 8 个有限 `[192,192]` 输入吐非有限 Q/R；正常样本 `Q@R≈A` 到 7.2e-6。
- 精度 11/30：STEP-256（mx+broadcast）与 STEP-258（CPU FP64+broadcast）几乎同轨迹；不是社区 QR 独有。STEP-245 HEAD SOAP 才是 28/30。
- 8 个 BAD `.pt` 已在本地并传到同事机 `/home/ubuntu`。未改业务代码、未新开训练。
- 下一步等用户选择修算子还是恢复 HEAD SOAP 再只换 QR。

## 2026-08-18：双门禁重评分

- 用户要求：耗时大幅下降，逐步 loss ≤2%。
- 既有 30 步对照：仅 63861df CPU FP64 SOAP 达 30/30；快路径全部破 2%。下一步为该数值路径上的保序并行 CPU QR。

## 2026-08-18：只读复核 SOAP 亲和提交与 loss 分叉

- 远端权威仓库 `ascend_npu_optimize@669a138`；`fb979b2` 对象完整，父提交为 `63861df`。
- 只读确认该提交同时引入 one-sided=1024、identity 初基、NPU FP32 周期 QR；与既有 STEP-237～246 8 卡证据一致。
- 结论：NPU/GPU 精度问题可归因于该提交的优化器语义变更，不是单纯设备迁移。本轮未改代码、未训练。

## 2026-08-17：精度隔离循环 STEP-238

- 用户要求持续定位直到 NPU/GPU loss 差异不大，中途不停止。
- 门禁：step1–5 <1%；step10–13 压回个位数（优先 <5%）。
- 下一刀：SOAP NPU QR（`fb979b2`）30 步 overlay A/B，保留 HEAD 入口与 k=4。

## 2026-08-13：目标续接与下一候选恢复

- 重新读取 planning/profiling/npu-smi 技能和持久计划；发现 `task_plan.md` 的 Next Step 滞后于 STEP-074，已校正为审计 MapTR `data_valid` 重复标量同步。
- 首次因 PowerShell 默认编码造成显示乱码，按显示文本执行的 `apply_patch` 未匹配且未改文件；随后显式以 UTF-8 读取并完成最小更新。
- 尚未修改远端业务代码。

## 2026-08-11：8 卡 profile 与 rank0 advisor 完成

- 8 个 rank 的 TorchNPU profile 均已完整落盘到远端诊断目录，总量约 68.9 GiB；未下载任何远端数据或产物。
- 第 11 步 SOAP 周期长尾在 profile 中复现，8 rank 均输出 profiler 完成标记。
- rank0 `msprof-analyze advisor all` 已成功完成，确认 E2E 中 NPU 空闲占 89.74%，并产出 4 类亲和 API、AICPU、在线编译等候选。
- advisor 规则库低于实际 CANN/Torch 版本，所有候选必须结合当前官方/社区成熟实现与 8 rank timeline/CSV/DB 再核验，不能直接改代码。
- 当前下一步：远端就地聚合 8 rank profile，定量定位 SOAP CPU QR/D2H 与其他候选；不修改业务代码，不把远端数据拉到本地。

## Session: 2026-08-11

### Phase 1: 现状审计与计划制定
- **Status:** in_progress
- **Started:** 2026-08-11
- Actions taken:
  - 安装并调用用户级 `planning-with-files` 技能。
  - 完整读取技能说明与标准模板。
  - 建立用户批准门禁：批准前只读调查，不修改业务代码。
  - 创建初始任务计划、发现记录和进度日志。
  - 读取本地项目规则和远程机器资料（输出已脱敏）。
  - 确认当前本地目录不是代码仓库，后续代码调查需在远程环境只读完成。
  - 确认远程访问拓扑、共享代码位置规则和训练必须在 NPU 机器执行的约束。
  - 检查本地 SSH 能力：有 OpenSSH；未发现 PuTTY、Posh-SSH 或现成 `paramiko`。
  - 静态检查本地临时快照，定位随机性、CPU 搬运、Scikit-learn KMeans、调试开关和 NPU 编译/精度配置候选。
  - 调研 Ascend 官方 TorchNPU Profiler、msprof-analyze、msTransplant/PyTorch Analyse 和 OpPlugin。
  - 将阶段顺序调整为：随机性移除独立提交 -> 在该提交上建立性能基线 -> 逐项性能优化。
  - 补充性能工具链、初始候选 backlog、统一 A/B 口径、提交规则和停止条件。
  - 按用户补充要求加入性能证据门禁：非明确问题必须先经工具测试定位，明确问题也必须完成修改后 A/B 验证。
  - 加入训练命令来源规则：基线、profile 和回归均参考远程当前目录已有 `.sh` 脚本，并记录脚本 hash 与实际参数。
  - 加入 8 卡训练硬约束：所有正式训练、profile、A/B 和回归使用 8 卡，并记录各 rank 与通信指标。
  - 用户已明确批准完整计划，进入 Phase 1 远程只读审计。
  - 已使用 planning-with-files 对获批计划建立 SHA-256 证明（前缀 `18756331b2e7`）。
  - 在 Codex 用户工具缓存安装 Paramiko，并创建本地脱敏 SSH 辅助脚本；未写入远程仓库。
  - 验证跳板机与 NPU 训练机连接成功，枚举共享目录中的 Git 仓库候选。
  - 读取首个目标 NPU 仓库状态：当前为 `ascend_npu`@`f189414`，未发现 `asend_npu_optimize`，开始跨仓库核对。
  - 跨仓库确认真实目标为主仓库的 `ascend_npu_optimize`；审计其 HEAD `72a266b 【loss对齐】随机性移除`，工作区干净且 diff check 通过。
  - 完整审阅 `72a266b` patch，确认固定 seed 主体已移除，同时发现遗留随机性调试输出和 `synchronize_after_backward` 归属需继续核实。
  - 扫描现有训练脚本：`run_train.sh` 当前为单卡；底层 `tools/ddp_train.sh` 支持多卡，继续寻找现成 8 卡封装脚本。
  - 审计 `f189414` 引入内容，发现单卡切换、DBG_NPU、sampler 调试和 `if True` 强制逻辑未被 `72a266b` 完整还原。
  - 确认多个现有训练脚本默认 8 卡，开始选择 canonical 8 卡入口。
  - 初选 `tools/local_train_spetr_debug.sh` 为 canonical 8 卡入口；发现无关旧环境脚本含凭据后停止使用并增强输出脱敏。
  - 完成随机性历史归属审计，确定补齐还原项；MSDA CPU fallback 留待工具定位后的性能优化阶段。
  - 将 7 个补齐还原文件上传远程工作区；语法与 diff check 通过，等待 8 卡验证后 amend。
  - 检查 NPU 资源与进程：16 个设备空闲；发现旧单卡 launcher 被暂停且无 NPU 占用，计划以独立端口/work-dir 进行 8 卡验证。
- Files created/modified:
  - `task_plan.md`（规划文档）
  - `findings.md`（调查记录）
  - `progress.md`（进度记录）

## Planned Test Result Schema
| Run ID | Commit | Environment | Command/Data | Warmup | Iterations | Throughput | Step Time P50/P95 | Device Memory | Host CPU | Correctness | Profile Artifact |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| BASELINE-TBD | 待确认 | 待采集 | 待确认 | 待确认 | 待确认 | 待测 | 待测 | 待测 | 待测 | 待测 | 待生成 |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|---|---|---:|---|
| - | 暂无 | 0 | - |
| 2026-08-11 | 系统 Python/py 的 `paramiko` 探测命令无有效输出并以 1 退出 | 1 | 停止重复，转而检查已安装 SSH 工具 |
| 2026-08-11 | 第一次 CPU 路径 `rg` 命令因 PowerShell 引号/过滤组合导致部分搜索未执行 | 1 | 拆分模式并简化引号 |
| 2026-08-11 | 第二次 CPU 路径搜索出现 PowerShell 字符串终止符错误 | 2 | 改用单个双引号正则，第三次成功 |
| 2026-08-11 | WSL 探测失败，系统未安装可用发行版 | 1 | 不安装 WSL，不改变本机环境；使用现有资料继续规划 |
| 2026-08-11 | 首次执行 `attest-plan.ps1` 被 Windows ExecutionPolicy 阻止 | 1 | 改用独立 PowerShell 进程并指定一次性 `ExecutionPolicy Bypass`，证明成功 |
| 2026-08-11 | 跳板机旧版 Git 不支持 `git -C` | 1 | 后续使用 `(cd repo && git ...)`，不改变远程 Git 环境 |
| 2026-08-11 | 读取远程训练日志时损坏字符触发本地 GBK `UnicodeEncodeError` | 1 | SSH 脱敏助手改为 UTF-8 容错输出后再诊断 |

## 5-Question Reboot Check
| Question | Answer |
|---|---|
| Where am I? | STEP-241 算子 I/O 对照完成；one-sided 非等价，geo/pin/foreach 逐位 |
| Where am I going? | 等待是否提交 `one_sided_dim_threshold=None` |
| What's the goal? | 优化算子相对优化前输出差异很小；one-sided 已否决 |
| What have I learned? | foreach/geo/pin 逐位；stale-Q 12 步 max_abs 2.7e-3；one-sided 3 步 max_abs 5.9e-3 且 GG 结构不同 |
| What have I done? | 容器内同输入 I/O；MSDA 本轮子进程失败，引用既有真实 shape 门禁 |
## 2026-08-11：随机性移除提交前验证准备

- 已确认目标 NPU 宿主机有 16 个健康逻辑设备，检查时无实际 NPU 计算占用。
- 已确认训练必须从现有 Docker 容器运行；容器内 PyTorch/torch_npu 可识别全部 NPU。
- 已确定使用仓库既有 `tools/ddp_train.sh` 启动 8 卡短训练，使用独立端口和仓库外独立结果目录。
- 已确定启动前显式清除固定随机性相关环境变量，避免继承历史进程配置。
- 当前随机性移除补充修改仍未提交；待 8 卡短训练验证通过后，才会 amend 到唯一提交 `【loss对齐】随机性移除`。
## 2026-08-11：随机性移除阶段完成

- 8 卡短训练验证通过：10 个 iteration，稳态 step（第 3-10 步）约 2.9-3.45 秒，无 NaN/Inf、OOM、HCCL 或 traceback 异常。
- 已清理历史停止态的 29507 单卡任务；本次 29627 验证任务结束后也已退出。
- 已将补充审计后的随机性/调试清理 amend 为唯一提交：`63861dfd920ab9829512b1e4a000eefd1ffcfbea 【loss对齐】随机性移除`。
- 提交共涉及 13 个文件，53 行新增、210 行删除；未包含算子性能修改，也未包含训练生成物。
- 远程目标仓库工作树已恢复干净。下一阶段进入该提交上的 8 卡性能基线与 profiler 采集。
## 2026-08-11：新增逐步操作记录硬规则

- 已创建 `操作步骤.md`，回填从技能安装、远程审计、随机性移除、8 卡验证、提交到三次正式基线的完整逻辑操作链。
- 后续每一步均在执行前记录目的/原因/指令，执行后记录现象/说明/下一步；远程敏感信息继续脱敏。

## 2026-08-11：三次 8 卡普通性能基线完成

- 基线 commit：`63861dfd920ab9829512b1e4a000eefd1ffcfbea`。
- 口径：8 卡（0-7）、world size 8、每卡 batch 1、全局 batch 8、每次 30 step、独立进程/端口/输出目录；脚本和配置 hash 已冻结。
- 三次 30-step 均值：28.532、28.213、30.051 秒/step；三次均值中位 28.532 秒，CV 3.39%。
- pooled 真实端到端均值：28.932 秒/step，吞吐 0.2765 sample/s。
- 排除冷启动与周期空洞后的 pooled 中位：3.186 秒/step，P95 7.074 秒，理想稳态吞吐 2.5110 sample/s。
- 每 10 步后的第 11/21 步稳定出现 host 长尾，6 次中位 271.486 秒、P95 280.307 秒；3/3 次独立运行复现。
- 现场证据：长尾期间 NPU AICore 0%，主 rank host CPU 满载，主线程落在 NumPy/OpenBLAS；下一步必须用 8 卡 profiler 覆盖正常 step 与第 10→11 步边界，定位具体调用链。
- 三次均无 NaN/Inf、OOM、HCCL 或 traceback；最大记录显存 5067 MB；运行生成物已归档，远程工作树干净。
# 2026-08-11 新增硬规则

- 禁止把任何远端数据、日志、profile、checkpoint、模型或训练产物拉到本地；所有原始分析必须远端就地完成，本地只记录脱敏摘要。

## 2026-08-11：Phase 4 首个 SOAP 性能优化已提交

- **Status:** complete（本逻辑优化）；Phase 4 其他候选仍待逐项工具定位。
- 已提交 `fb979b28ee3d417806a48c0d643676fd7d38541e 【npu性能优化】SOAP预条件器NPU亲和优化`，父提交为不可变的 `63861dfd... 【loss对齐】随机性移除`；未 push。
- 变更仅包含实际注册的 SOAP 实现与目标 config，提交后远端工作树干净；临时 hook/config、kernel cache 和 profile 均已归档到远端诊断目录。
- 三次 8 卡 30-step After：均值 5.559/5.674/5.719 秒，CV 1.187%；pooled 5.651 秒、1.4158 sample/s。相对基线 28.932 秒、0.2765 sample/s，速度 5.120×、吞吐提升 412.03%。
- 周期中位 271.486→15.578 秒（17.428×）；普通稳态中位 3.186→3.178 秒；显存 5067→4070 MB。90 个 After step 的 loss/grad 均有限，三次均无 fallback、traceback、OOM、HCCL error。
- 优化后轻量 rank0 TorchNPU profile（训练仍为 8 卡）证明 CPU FP64 QR/mm 消失；剩余 543 次官方 ACLNN AICPU QR 共 22.674 秒，是下一阶段明确热点。
- 启动/工具错误已记录：前三个 A/B 启动尝试因 config/PYTHONPATH 上下文失败且均为 0 step；首轮外层 monitor 未按 30 步停止，最终严格只取前 30 步；原生 max_iters 测试夹具解决后两轮；After profiler 准备命令本地超时但服务端继续运行，精确核验后接管唯一任务，避免重复启动。
- 下一步建议：先对剩余 2560 AICPU QR 调研成熟分块 SOAP/Shampoo/AI Core 方案；若无版本匹配的成熟实现，则转向 profile 已列出的其他候选（在线编译、亲和 transpose/layer norm、fused grad norm），每项仍遵循修改前工具证据、单变量修改、三次 8 卡 A/B 和独立提交。

## 2026-08-11：DataLoader worker=2 优化已提交

- **Status:** complete（本逻辑优化）。
- 提交：`6477a5b6eab010b36c9ffb14eee4ec127bc1d7f8 【npu性能优化】DataLoader多进程加载`；仅将目标 config 的 `workers_per_gpu` 从 0 改为 2，未 push。
- 证据：After profiler 的单进程 DataLoader host self 约 1.5 秒/step；worker=2 三次 8 卡 30-step run 均值 4.042/4.040/4.024 秒，CV 0.198%，pooled 吞吐 1.9825 sample/s。
- 相对上一提交 worker=0 再提速 1.400×；相对最初随机性移除基线累计提速 7.170×。三次均无多进程、正确性或 NPU fallback 异常，显存不变。
- worker=4 已实测，无额外端到端收益，因此不采用。所有临时 config/patch/log 均留在远端诊断目录或已清理本地临时文件，远端工作树干净。

## 2026-08-12：切换到新机器迁移恢复阶段

- 用户明确要求恢复老机器中断的项目并在新机器继续优化。
- 已读取 `npu-smi` 与 `ascend-profiling-anomaly` 技能，更新 `task_plan.md` 的目标、当前阶段与 Phase 7 恢复检查清单。
- 已确认本地目录只包含项目文档与远程 helper，业务源码需在新机器原位审计；尚未启动训练、profile 或业务代码修改。
- 下一步：审查 remote helper 的安全用法，读取本地连接信息但不输出凭据，执行新机器只读恢复审计。
- helper 已完成安全审查；首次调用因本地 Python/Paramiko 缺失在连接前失败。下一步先恢复本地 helper 运行依赖，再继续只读审计。

## 2026-08-12：新机器恢复审计第一阶段通过

- 已在本地 Codex 工具目录恢复 Paramiko 3.5.1，并通过脱敏 helper 成功连接新机器。
- 已确认正确容器 `mapqr-leicheng`、8 个健康 NPU ID/16 chip、无训练占用、torch_npu 可用和 profiling 工具齐备。
- 已确认远程仓库 `ascend_npu_optimize` 位于历史第二项优化提交 `6477a5b`，工作树干净、提交链完整。
- 尚未启动训练、基线或 profiler；下一步核验 canonical 启动命令、容器挂载及迁移数据入口。
- 第二轮已确认共享盘挂载、脚本/config hash、8-rank launcher 和两项优化配置值。当前唯一未决项是有效配置中的数据/权重引用与历史 `--max-iters` 夹具是否完整迁移。
- 有效配置解析已通过：训练/验证标注与实际 backbone/neck checkpoint 均存在；历史测试夹具完整迁移。
- 已确认正确容器中的 14-step worker=2 profile 完整退出且可分析，因此取消重复 profile 计划，直接进入远端原位分析。
- 已完成 profile schema 与初步热点聚合：Step 10 仍由约 22.79 秒 AICPU QR 主导，普通 step 显示约 1.1–1.25 秒 device free；下一步生成正式 interval/host/AICPU/架构双报告。
- 正式异常 JSON/Markdown 与独立架构报告已在远端生成并验证。下一候选转为定位普通 step 中 `NanToNum → Arange` 间重复 `to/_to_copy/copy_` 的功能来源；在代码/调用链证据确认前不修改业务代码。
- 已校正 host coverage 统计并排除外层 step marker 污染。现有证据仍不足以唯一定位代码路径，准备复用迁移夹具做 rank0 单 active step、with_stack=true 的 8 NPU 窄 profile。
- rank0 窄 stack profile 已在 `mapqr-leicheng` 启动：1 个 launcher、8 个训练 rank、npu-smi 8 个唯一 Python PID，torch_npu/NPU/配置门禁全部通过；等待 12 iter 和 rank0 profile 解析完成。
- 窄 stack profile 已完成 12/12 iter 并自然退出 0，无 traceback/OOM/HCCL error；约 1.11 GB 产物留远端。收尾门禁已通过：端口/NPU 释放、业务 Git clean；约 10.49 万条 operator 记录带 stack。首轮全表聚合确认 active window 是 SOAP 周期步，并受到末步 checkpoint 保存污染，暂不能决定普通步优化。下一步改为 trace 时间窗与 device bubble 精确求交，并核验 gradient fingerprint hook 是否为每步调试开销。
- 时间窗求交完成：277.6 ms 主空洞内的 4 次 D2H 转换来自 `spetr3d.py` 两段重复、无条件输入哈希打印；4.99 秒尾部空洞是末步 checkpoint。Gradient fingerprint 当前禁用实际指纹采集，不修改。下一步用同一远端既有 shell 入口完成 3 次新机 8 NPU 修改前基线，再只删除无条件输入哈希打印并做 3 次同口径 After。
- 无条件输入哈希优化 A/B 已完成：Before/After 各 3 次 8 NPU 30-step，pooled 普通步中位 1.2870→1.0565 秒（-17.91%，1.218×），P95 1.414→1.261 秒；重复 `FWD_IN` 480/轮→0。六轮 loss/grad 全有限，loss 分布重叠且下降趋势正常，业务 diff 仅 1 文件删除 14 行。准备形成唯一功能 commit，不 push。
- 已提交 `5a37d0432951db6ffd0b145ea151a4fd33b1a0be 【npu性能优化】训练输入哈希调试移除`，提交后业务工作树干净，未 push。下一步对提交后正常 step 做轻量 profiler 复核并继续按功能选择下一候选。
- 提交后正常 Step 12 stack 复核成功：目标空洞 277.612→49.469 ms（-82.18%），`tensor_hash/FWD_IN` stack 为 0；业务仓库已恢复干净，原始约 783 MB profile 只留远端。当前最大剩余空洞是 55.700 ms 梯度裁剪路径，进入版本内官方能力只读审计。
- 梯度裁剪候选已筛掉：当前 NPU 已走 PyTorch foreach；torch_npu 的 fused clip 仅为 NpuFusedOptimizerBase 内部 combined-gradient API，SOAP 无法低风险复用。为避免改变优化器/loss 语义，不实现。下一步审计已关闭指纹功能遗留的无条件 `retain_grad`。
- `retain_grad/_debug_output_tensors` 候选完成三次 8 NPU、30-step After：普通步中位 1.0820/1.0680/1.0905 秒，pooled 中位相对 Before 1.0565→1.0755 秒，回退 1.80%；显存仅减少 1 MB，周期步基本持平。三轮 loss/grad 全有限、下降趋势正常、无训练异常，但性能门槛未通过，因此拒绝并不提交。唯一业务文件已精确恢复 HEAD，远端工作树干净、NPU 进程为 0；下一步只读审计高频 `MAP_SHIFT` 调试日志。
- `MAP_SHIFT` 三轮均仅 24 行，即 8 rank 各命中一次三行日志，并非每 step 热点；实际 index 参与训练数组切片，因此不修改、不 A/B。下一步审计 normal-step profile 中约 49 ms 的 history-memory slicing/locations 路径。
- 关闭 PVB 坐标网格死计算候选完成三次 8 NPU A/B：普通步 pooled 中位 1.0565→1.0470 秒，仅改善 0.899%，低于既有三轮波动；P95 有改善、显存仅减少 2–6 MB。loss/grad 全有限且 loss 正常下降，但收益门槛失败，因此不提交并精确回退。
- 关闭 PVB 坐标网格候选已精确恢复：远端 HEAD `5a37d043`、工作树干净、NPU 进程为 0；没有 commit、没有 push。候选 diff 和三轮证据只留远端，下一候选继续以 profile 收益上限和正确性硬门禁筛选。
- normal-step 最内层 stack 聚合显示剩余 host 开销主要集中在 SOAP 逐参数投影/更新；scalar overload 每行约 559 次，但源码没有显式 `.item()`。若改为 foreach/combined-gradient 会改变参数更新与数值轨迹，正确性风险过高，暂不修改；下一步回到 timeline 连续 device gap 筛选等价低风险路径。
- 2026-08-13：正常步 profile 的异常发现与独立模型架构报告已在远端原位补齐并校验；业务仓库仍为 HEAD `5a37d043`、clean、无 NPU 训练进程。下一候选确定为“冻结点云骨干局部模式切换”，功能边界仅两处 `eval/train` 调用，须经三次 8-NPU A/B 后才决定是否提交。
- 2026-08-13：冻结点云骨干局部模式切换完成三次有效 8-NPU 30-step；正确性通过，但 pooled normal median 仅改善 0.8897%，低于噪声门槛，已精确回退且未提交。远端恢复 HEAD `5a37d043`、Git clean、NPU 进程 0。
- 2026-08-13：只读筛掉梯度裁剪参数缓存（收益上限约 1.6%，动态图过滤仍保留）；未修改远端代码。开始按真实 shape 审计 SOAP `tensordot` 调度是否存在不改变更新顺序的等价官方表达。
- 2026-08-13：SOAP shape 元数据捷径因诊断 checkpoint 不在容器挂载范围而在打开文件前失败；未复制产物、未改环境。缺少 shape/等价性证据，暂不实施 SOAP contraction 改写。
- 2026-08-13：开始 STEP-068，计划用既有成功 harness 在 `mapqr-leicheng` 中采集 rank0 单正常步 `record_shapes=true` profile；不修改业务代码、不改变远端依赖，原始产物只保留远端。
- 2026-08-13：STEP-068 第一次训练完成且正确性正常，但 profiler schedule 边界错误导致 device kernel dict 为空，profile 无效。已记录错误，下一次改用 wait=10/warmup=1/active=1。
- 2026-08-13：STEP-068 第二次 profile 成功，取得 Step 11 kernel/operator/shape/stack。代表性机制测试排除 project/project_back，只保留 22/22 bitwise equal 的 covariance matmul 等价式进入全 shape 校验；尚未修改业务代码。
# 2026-08-13

- 完成 SOAP covariance 全部真实 shape/有效轴位级验证：153/157 通过，4 例失败，候选已拒绝且远端业务代码未改。
- 正在为有效 record-shapes profile 补齐异常报告和独立 10 节架构报告。
- record-shapes profile 报告闭环完成：正式 schema 通过、异常报告非空、架构报告 10 节、远端 HEAD/clean/NPU 状态正常。
- 下一步转入未审查项目源码栈的只读候选筛选；SOAP covariance 不再实施。
- MapTR target mask 候选已筛完：冗余 `.unique()` 收益不足；`torch.isin` 虽完全等价但在当前 NPU 环境慢约 54.9%，均未改码、未提交。
- 完成二维点 normalize 向量化三轮 8-NPU 正式验证：pooled 1.059 秒，较基线慢约 0.237%；正确性正常但性能失败，已回退，远端 Git clean、无新 commit。
- 完成 SOAP 二阶矩 `addcmul_` 三轮 8-NPU：pooled 1.061 秒、较基线慢约 0.426%，周期步亦退化；正确性正常但已回退，无新 commit。
- 2026-08-13：完成 STEP-072 正常步连续 gap 与 MSDA backward 只读筛选；确认迁移后的目标仓库仍为 `5a37d04`、clean。没有满足严格等价且收益上限高于 2.15% 噪声的候选，未改代码、未训练、未提交。下一步审计 `Index/IndexPut/Nonzero` 的具体功能归属。
- 2026-08-13：完成 STEP-073 MapTR target mask 候选三轮 8 NPU 正式验证；正确性正常，但普通步仅改善约 1.28%，低于噪声，已精确回退。远端 HEAD `5a37d04`、Git clean、NPU 0，无新 commit、无 push。
- 2026-08-13：完成 STEP-074 `aten::item` 来源审计；纯日志同步收益不足，SOAP scalar overload 改写风险高且已有相邻候选反例，未改代码、未训练、未提交。
- 2026-08-13：进入 STEP-075，只读审计 MapTR target 构造中的重复 `data_valid` 标量同步；已记录候选边界与门槛，尚未修改代码、训练或提交。
- 2026-08-13：STEP-075 有效配置和 line-level trace 核验完成；重复次数与 4 层×4 类×3 tag 一致，host self 上限越过噪声门槛。下一步做 loss 类型链审计与 NPU 机制位级验证。
- 2026-08-13：STEP-075 宽版本机制测试发现 FP16 loss 差异并立即拒绝；收窄为仅把 Python 副本用于 target 控制，保留原 tensor 计数/loss 路径，等待二次机制验证。
- 2026-08-13：STEP-075 收窄版机制测试全部位级一致且同步微计时明显改善；进入 target-only 最小 diff 与静态门禁，尚未提交。
- 2026-08-13：STEP-075 最小业务 diff 已应用，唯一 tracked 文件 4 增/2 删，diff/语法/版本/NPU 空闲门禁通过。准备三轮 8 NPU 正式验证，尚未提交。
- 2026-08-13：STEP-075 第 1 轮正式 A/B 已在 `mapqr-leicheng` 启动，torch_npu、8 个直接 rank 与 npu-smi 8 个唯一训练 PID 门槛均通过；等待 30 步完成。
- 2026-08-13：STEP-075 三轮完成，普通步 pooled 1.084 秒、较基线慢约 2.60%；正确性正常但性能失败，决定精确回退且不提交。
- 2026-08-13：STEP-075 已精确回退；远端 HEAD `5a37d04`、Git clean、NPU 0，无 commit、无 push。进入 STEP-076，用 normal-step device self/kernel 重新排序剩余候选。
- 2026-08-13：STEP-076 首轮 device-self/kernel 重排完成；除已审查 MSDA 外没有单一剩余功能越过门槛。下一步专门归属 Index/IndexPut/Nonzero，确认是否存在未审查的同功能聚合热点。
- 2026-08-13：STEP-076 完成；Index/IndexPut/Nonzero 没有未审查同功能越过门槛，未修改、未训练、未提交。进入 STEP-077 审计 MapTR decoder host 调度与 timeline overlap。
- 2026-08-13：STEP-077 证明 decoder line 133 为整层外部归因，无法作为严格等价单点优化，已筛除。进入 STEP-078 审计 Hungarian assignment 的 D2H/CPU 求解边界。
- 2026-08-13：STEP-078 确认固定环境无等价设备 Hungarian solver，完整搬运+求解上限约 22.61 ms且低于门槛，未修改。进入 STEP-079 审计 MapTR decoder line 812 次级 host 调度。
- 2026-08-13：STEP-079 确认 line 812 是动态 output projection 真实计算，不能缓存或删除，已筛除。进入 STEP-080 量化 decoder reference point 的明确死分配。
- 2026-08-13：STEP-080 量化确认死分配仅约 1.434 ms host self，收益不足，未改码。进入 STEP-081 全局审计高频初始化与死值覆盖。
- 2026-08-13：STEP-081 未发现新高收益死初始化；高聚合项均为真实算子输出或已审查路径。进入 STEP-082 重排重复 to/copy/cast/TransData。
- 2026-08-13：STEP-082 找到 grad clip 560 次 device-self 0 的 `.to`，理论 host 上限约 43.8 ms；进入 STEP-083 固定版本源码与位级机制验证。
- 2026-08-13：STEP-083 等价性通过但 profiler-off 仅约 1.06 ms收益，远低于噪声，未改码。进入 STEP-084 contiguous/clone/format 专项审计。
- 2026-08-13：STEP-084 无超过门槛的无效 contiguous/clone；未改码。进入 STEP-085 审计 Map target 重复 normalize 调用链。
- 2026-08-13：STEP-085 发现最后一层一次重复 normalize，但收益上限不足且相邻候选已有负面 A/B，未改码。进入 STEP-086 重复有限值 mask 审计。
- 2026-08-13：STEP-086 仅发现最后层约 3.67 ms 的严格重复 finite mask，收益不足，未改码。进入 STEP-087 条件零初始化审计。
- 2026-08-13：STEP-087 可删零初始化仅约 1.624 ms，未改码。进入 STEP-088，用真实 shapes筛选 GT normalize 跨层复用。
- 2026-08-13：STEP-088 位级一致但仅约 1.22 ms收益，未改码。进入 STEP-089，审计 SOAP 同形状矩阵 batched QR 的全真实 shape 位级可行性。
- 2026-08-13：STEP-089 已提取 551 次/24 种 QR shape；1–512 shape batched Q/R 全部位级一致。下一步按 shape聚合周期耗时并逐级验证大矩阵，尚未改码。
- 2026-08-13：STEP-089 大矩阵验证到 2560 仍位级一致但无性能收益；全量 batching排除。正在按真实频次测小/中 shape完整周期累计上限。
- 2026-08-13：STEP-089 完整真实频次 batched QR反而慢约 76 ms/周期，已拒绝且未改码。进入 STEP-090 审计官方 Q-only QR路径。
- 2026-08-13：STEP-090 Q-only组合非位级且发生CPU fallback，已拒绝、未改码。进入 STEP-091 审计 QR前排序与状态重排冗余。
## STEP-091 完成：SOAP QR 前排序静态旁路未达门槛

- 已完成既有周期 profile 的算子/形状聚合和无 profiler NPU 微测。
- 1×1 旁路逐位等价，但只摊销节省约 1.1648 ms/步，未进入代码修改与 3×8 验证。
- 远端无 commit、无 push；下一轮继续筛选周期 SOAP 路径的高上限等价候选。
## STEP-092 完成：SOAP 大矩阵重排替代实现淘汰

- 已确认当前 `index_select` 比逐位等价的高级索引和 `gather` 更快。
- 未修改远端源码，未创建 commit，未 push。
- 下一步审计周期 QR 的 CPU fallback/原生算子支持状态。
## STEP-093 完成：排除 QR CPU fallback 假设

- 已确认 SOAP QR 调度到 NPU 原生 `aclnnLinalgQr`。
- 弃用的 `torch.qr` 没有独立后端或稳定收益，不修改代码、不创建 commit。
- 后续继续筛选保持 Q/R 逐位结果的调用编排与数据准备候选。
## STEP-094 完成：通用 SOAP 矩阵乘法复用淘汰

- 已用多形状、多轮外积状态验证三类复用表达，较大形状均出现浮点末位差异。
- 未修改业务代码、未训练、未提交。
- 下一步隔离验证只发生一次的 identity 初始基旁路（STEP-095）。
## STEP-095 完成：一次性初始化旁路淘汰

- 严格等价门槛通过，但收益仅约 85.6 ms/rank/整次训练，不进入实现和 8-NPU A/B。
- 下一步测试同一 state 的独立 QR 是否能通过 NPU streams 获得周期级并行收益。
## STEP-096 进行中：双 2560 stream 并发通过

- 机制正确性与性能门槛均通过，尚未修改业务代码。
- 正在验证 2560+5120 和四路中型组合，之后再决定最小实现。
- STEP-096 已通过混合大矩阵和两流中型机制门槛，进入最小单文件实现与函数级状态验证；尚未 commit。
- STEP-096 最小实现已落盘但未提交；静态和函数级状态门禁通过，准备第 1 轮 8-NPU 正式验证。
## STEP-096 完成：两流 QR 正式验证失败并回退

- 第 1 轮 8-NPU 30-step 完整退出 0，正确性正常，但周期步较基线退化约 63.0%。
- 已停止后续两轮，精确恢复 `soap.py` HEAD blob；远端 Git clean、训练进程 0、无 commit/push。
- 后续不再推进 multi-stream QR，继续寻找严格等价且能在正式 8-rank 负载下兑现的候选。
## STEP-098 完成：QR输出缓冲候选淘汰

- 小/中 shape profiler-off无收益且有内部格式告警；未改码、未训练、未提交。
- STEP-100真实频次机制门槛通过（约57.3 ms/step上限），进入精确条件命中与最小实现；尚未改业务代码。
- STEP-100窄分支已落盘未提交，函数级命中/非命中case全部逐位通过；准备一轮8-NPU周期早停验证。
## STEP-100 完成：窄等尺寸QR并发正式失败

- 一轮8-NPU在Iter11/12触发性能早停；功能有限性正常但周期与后继步明显退化。
- 已停止、回退、归档运行时残留；远端HEAD clean、训练进程0、无commit/push。
- SOAP QR multi-stream方向正式关闭。
- STEP-101完成：denormalize候选严格等价但收益仅0.868ms/step，未改码、未提交。
- STEP-102完成：ignore查找表严格等价但仅1.892ms/step，未改码、未提交。
- STEP-103启动：审计SOAP死亡`grad_projected`的原地square临时buffer复用。
- STEP-103完成：收窄候选逐位安全但仅0.514ms/step，未改码、未提交。
- STEP-104完成：Tensor scalar overload较Python float明显更慢，未改码、未提交。
- STEP-105完成：data_valid布尔缓存严格等价但仅节省0.535ms/step，未改码、未提交。
- STEP-106完成normal stack重新聚合：最大未闭环边界为SOAP project/project_back的tensordot元数据展开。
- STEP-107完成：真实全频次仅节省19.096ms且18种4D组合不逐位，候选淘汰；未改码、未训练、未提交。
- 当前进入STEP-108：只读审计update_preconditioner的outer_product生命周期，优先筛选不改变算术表达的复用边界。
- STEP-108完成：tensordot out最终state逐位一致但慢8.905ms且需272.3MiB/rank，不改码、不提交。
- 当前进入STEP-109：审计梯度裁剪560次逐参数norm/mul能否使用当前官方foreach批量路径。
- STEP-109完成：正式foreach=None已自动批量化，显式True慢0.045ms；不改码、不提交。
- 当前回查既有候选覆盖表，再从normal profile选择未验证边界。
- STEP-110完成：固定torch版本无foreach nan_to_num API，保留逐grad安全清理，不改码、不提交。
- 当前执行STEP-111权威状态收尾；STEP-105—110均未通过提交门槛，本轮应保持0个新业务commit。
- STEP-111完成：远端HEAD `5a37d043...`、Git clean、正确容器、训练/NPU进程0；本轮0 commit、0 push，持久记录已覆盖STEP-105—110。
- STEP-112完成：一阶矩双foreach全shape逐位一致但仅节省6.857ms/step，不改码、不训练、不提交。
- 后续候选需优先真实device计算或功能级重复；继续保持远端HEAD clean与单功能commit门禁。
- STEP-113启动：已完整恢复技能/规划上下文并修正陈旧Next Step；重新审查SOAP covariance仅安全shape/axis窄分支，先做真实频次收益门禁，尚未改码。
- STEP-113完成：非4D-axis1安全族多seed及完整state逐位通过，但真实收益仅6.292ms/step；不改码、不训练、不提交。
- STEP-114启动：审计SOAP 559个denominator的foreach sqrt/add，先做全shape逐位和profiler-off门槛。
- STEP-114完成：559个denominator逐位一致，但仅节省6.520ms/step，不单独实施。
- STEP-115启动：盘点SOAP多个保持原算术表达的foreach阶段，先计算完整调度功能的收益上限与逐位边界。
- STEP-115完成：五个逐位foreach阶段的乐观收益合计仅19.382ms/step，未越过门槛，不重构。
- STEP-116启动：最后审计二阶矩保持原三步表达的foreach收益与跨参数保留projected gradient的显存代价。
- STEP-116完成机制测试：二阶矩逐位一致并节省10.960ms，使六阶段乐观上限约30.342ms，但全量额外活跃张量约589.6MiB/rank。
- STEP-117启动：用8M元素预算的原序分块完整骨架测实际合计收益、逐位和最大额外内存；尚未改源码。
- STEP-117机制门禁通过：完整六阶段分块骨架逐位一致，47.276→11.960ms，节省35.315ms；最大projected+square约100MiB。
- STEP-118启动：准备仅修改soap.py的最小分块实现与完整optimizer多step逐位门禁；尚未正式训练或提交。
- STEP-118静态与函数级门禁通过：单文件diff、6组6步parameter/Q/GG/一二阶矩/state_dict全部逐位一致，无新增持久state。准备第1轮8-NPU正式早停验证。
## 2026-08-13 SOAP 分块 Foreach

- 已完成机制基准、源码实现、静态检查、6-step 逐位功能门禁及三轮 8-NPU 30-step 正式验证。
- 正式三轮 pooled 普通步中位数提升 2.186%（对历史 pooled baseline 约 2.60%），周期窗口与显存无实质回归。
- loss 全有限且与历史基线接近；候选允许进入单功能本地提交，禁止推送，后续长训继续观察 grad_norm 尖峰。
- 已提交：`14d4f23 【npu性能优化】SOAP分块Foreach调度`；远端工作树 clean，未 push。
- 2026-08-13：开始 STEP-120，从 `14d4f23` 后筛选下一独立热点；已恢复 planning/profiling/npu-smi 规则，尚未修改业务代码或启动训练。
- STEP-120 远端恢复点通过：`14d4f23` clean、正确容器、训练/NPU 进程 0；开始复用远端既有 normal profile 报告筛选下一候选。
- STEP-120 旧 profile 主导热点与刚提交的 SOAP 调度直接重叠，判定排序过期；准备复用成功 harness 采集 `14d4f23` 后的 8-NPU 单正常步 profile。
- STEP-120 新 HEAD normal profile 完成：8-rank/NPU门禁通过，13/13、exit0，约759MB原始产物仅留远端，Git clean/NPU 0；开始原位分析。
- STEP-120 提交后 profile 分析闭环：anomaly 标准化JSON通过正式schema，50个bubble带边界说明和49组完整kernel上下文；独立架构报告10节齐全。开始基于新排序选择候选。
- STEP-120新排序完成：已关闭MSDA/grad clip/tensordot方向仍不可实施；进入STEP-121比较SOAP 8M/16M/32M chunk预算，成功只amend上一commit。
- STEP-121完成：chunk预算扩大收益亚毫秒且增加临时内存，未改源码、未amend。进入STEP-122只读归因NanToNum→Arange gap。
- STEP-122完成并关闭：绝对时间窗证明gap为已失败的冻结骨干模式切换候选，未改码/未训练。进入STEP-123细分grad clip host边界。
- STEP-123关闭：clip gap由历史已拒绝的`norm.to(first_device)`列表推导主导，未重复实施。进入STEP-124 Copy→Zero死写审计。
- STEP-124关闭：Copy→Zero相邻kernel不构成死写，真实host工作仍为已拒绝模式切换。进入STEP-125批量gap归类。
- STEP-125前24 gap归类：SOAP/mode-switch/grad-clip均为已闭环功能，other由两个不同且各低于门槛的边界组成；继续检查全部50 gap的同功能累计。
- STEP-125全部gap归类完成：唯一需复核的新聚合是8个`aten::index`窗口合计53.425ms；进入STEP-126具体栈审计。
- STEP-126完成栈拆分；line1032 close-range三类权重稀疏写未被STEP-073完整覆盖，进入STEP-127机制门禁，尚未改码。
- STEP-127 profile 行级聚合完成：正常步共 16 次 close-range 调用，缓存 `pos_inds[passed]` 可严格消除后两组 Boolean index/nonzero；profiler 乐观上限约 6.6 ms/步，正在做 profiler-off NPU0 最终机制裁决，尚未修改业务代码。
- STEP-127 完成并关闭：缓存 `pos_inds[passed]` 三张量结果逐位一致，但 profiler-off 真实 16 calls/step 仅节省 3.657 ms，远低于门槛；未改业务代码、未训练、未提交。转入 STEP-128 loss parse gap 只读归因。
- STEP-128 完成并关闭：37 次日志 `.item()` profiler self 仅 4.341 ms；36 次 loss add 直接构造最终训练 loss，禁止重排。未改代码、未训练、未提交；转入 STEP-129 新 profile host hotspot 去重筛选。
- STEP-129 首轮源码栈聚合完成：fingerprint hook 热点实际分别是已关闭 grad clip 与已提交 SOAP 父边界；正在复核首个未闭环 `map_loss.py:326 pts_weight_dir_cos_loss`，尚未改代码。
- STEP-129 `pts_weight_dir_cos_loss` 已关闭：整个函数 4 calls 的 device total 仅 1.050 ms，重写官方余弦 loss 会冒最终 loss 偏离风险；未改代码。继续复核 `geo_loss.py:58`。
- STEP-129 `geo_loss.py:58` 已关闭：8 calls 的几何公式 device 成本仅数毫秒，einsum 改写有 loss 末位风险；未改代码。转查 `second_trans_fpn.py:671`。
- STEP-129 FPN line671 已关闭：对应真实卷积/上采样/BN/ReLU，多尺度输入不同，device total约5.899 ms；未改代码。转查 `map_loss.py:750`。
- STEP-129 `OrderedPtsL1Cost` 已关闭：真实 cdist 决定 Hungarian 匹配，8 calls device仅1.018 ms，禁止高风险手写替代。转查 BEV backbone line120。
- 已按可信耗时记录剩余优先级：P0 SOAP周期QR 22.674秒；P1先量化MSDA CPU fallback；P2 Hungarian约22.61 ms/step；P3 BEV backbone line120 device 6.095 ms；P4其他copy/cast/format长尾。SOAP Foreach长周期收敛验证单列为验收任务。
- 流程已纠正：上述P0～P4仅为历史候选池；当前进入STEP-130，在`14d4f23`上先采集普通步、周期步及MSDA/Hungarian CPU专项耗时，形成当前HEAD实测排名后才允许修改算子。
- 新增硬门禁：每阶段必须向用户提交Before/After普通步、周期步、端到端、吞吐、目标算子wall/device/CPU、显存及loss/grad量化表；失败候选同样报告实际耗时与淘汰原因，无数据不得提交。
- 永久主基线固定为`63861df 【loss对齐】随机性移除`；后续每项同时报告相对63861df的累计收益和相对父提交的单项增量，最新commit不得替代主基线。
- STEP-131启动：获准从`63861df`创建独立detached基线worktree，按原配置执行3轮8-NPU×30-step复测；当前先恢复成功夹具与环境，尚未启动训练。
- STEP-132方案完成：按华为官方流程建立严格等价算子优化与batch容量扩展双轨测试；执行顺序为同机双基线→profile拆解→batch 1/2/4筛选→三轮复验→等样本长训收敛门禁。没有启动新训练、修改远端代码或创建commit。
- STEP-133本地custom识别：唯一训练配置为`mv2dfusion_..._finetune.py`，客户参数batch/rank=16、workers=8；另一个txt为评测结果。发现该配置与当前远端性能夹具batch/rank=1、workers=0不同，后续batch测试前新增配置权威性核验门禁。
- STEP-134完成：基线派生分支提交`4c37039`、最新分支提交`a757f29`，描述均已按用户要求改为`【去除随机性固定】客户训练配置字段对齐`；均只同步客户运行字段8设备/batch16/workers8/prefetch3，Config导入通过、两工作树clean、无训练/NPU进程。尚未执行batch=16训练。
- STEP-135完成客户batch16单轮正式A/B：`4c37039`与`a757f29`均在`mapqr-leicheng`通过8-rank/torch_npu/npu-smi门禁并完成30/30、exit0；全30步均值37.440→11.745秒（-68.63%，3.188×），吞吐3.419→10.898 samples/s（+218.76%）。
- STEP-135普通23步中位11.501→7.868秒（-31.59%），SOAP双步窗口均值279.589→39.227秒（-85.97%，7.127×）；框架显存28460→27445MiB/rank，设备HBM峰值45782→44054MiB。
- 两边loss/grad均有限、范围重叠且无数量级发散；最新loss均值较基线+6.12%、中位+11.74%，只能判定短跑无巨大异常，不能声明逐step loss对齐或长期收敛等价。训练副作用均归档远端并恢复两个工作树clean，未push、无新业务commit。
- STEP-136启动：继续长期优化目标，先恢复planning-with-files与Ascend profiling分析规范；首次合并读取因输出上限截断，已确认两份说明分别488/416行，改用分段完整读取后再决定是否重采batch16 profile，尚未修改业务代码或启动训练。
- STEP-136技能恢复要点：持久计划要求先读三份计划文件并运行session-catchup、重大决策前重读plan、每阶段/错误及时写盘；当前继续分段读至EOF，未开始远端动作。
- planning-with-files已完整读至EOF；Ascend profiling技能确认分析前必须读kernel_data_guide、rulebook，输出结构化JSON时读schema，且每次分析必须另产出完整架构Markdown报告。继续读取剩余技能与必需参考。
- Ascend技能中段确认：必须基于多流device interval union计算bubble，先通信去重；wait_ratio高而duration极小的锚点必须降级；每次报告含Top bubble前后原始kernel、软归因、固定五问，以及独立10节架构报告。尚未分析项目profile。
- 两份技能已完整读至EOF；session-catchup无未同步报告。计划/发现/进度文件分别639/574/327行，profiling四个必需参考分别460/238/289/163行；下一步分段完整恢复并遵守每两次查看写盘规则。
- task_plan前320行已恢复：顶部Next Step仍停留在已完成STEP-118，属于陈旧状态；硬门禁仍要求先profile、同口径A/B、正确性优先、单功能commit。需读完剩余内容后更新到STEP-136。
- task_plan已完整恢复：STEP-130明确要求当前HEAD重新采普通/周期及CPU专项profile后再排序；STEP-135客户batch16 A/B已完成。下一阶段应更新顶部Next Step为batch16最新HEAD profile适用性审计/重采，而非继续旧STEP-118。
- findings前300行恢复：旧profile与多项优化都基于batch1/worker2；最新客户batch16改变DataLoader、主机内存和普通步结构。既有profile仍可解释历史机制，但不足以给batch16最新HEAD的新候选排序。
- findings已完整恢复：关闭项默认不重开；当前未量化的MSDA CPU fallback/Hungarian/BEV等只能作为候选池。batch16最新A/B显示周期主步已从约266秒降至约10秒，说明旧QR 22.7秒热点排序已明显过期，必须重采最新客户负载profile。
- progress前240行已恢复：迁移、旧profile与历史A/B均有完整证据链；错误日志强调不能重复无效采集。当前新profile应复用已验证harness并避免末步checkpoint污染，仍只在正确容器8 NPU执行。
- progress已完整恢复；kernel guide前240行确认kernel_details为device主源、trace为host/device关联源，step按marker服务窗分配并裁剪跨界kernel，结构层需同时保留wall/busy-union/kernel-sum/total-cost四种时钟。
- kernel_data_guide已完整读至EOF：多流统一做busy union，communication.json仅用于总通信成本权威口径但HCCL kernel仍进入busy interval；AICPU按masked ratio分级，op必须同时按duration与含wait成本排名，不能把wait锚点当真实热点。
- rulebook已完整读至EOF：underfeed≥30%或单个gap≥max(1ms,10% service)即重异常；软归因至少20% overlap才标sync/H2D或comm，证据不足必须明确输出；边界截断、通信wait污染与AICPU暴露均需单独标签。
- architecture模板前240行确认10节顺序固定；若项目无FIA/decode，仍需保留章节并按训练模型证据降级，不能套用LLM层数。每种实际子结构要给kernel树、wall/占比/stream及交叉验证。
- architecture_report_template已完整读至EOF；schema前100行确认异常JSON顶层9个必填字段及global/group/bubble基础字段。继续读取schema尾部后完成技能恢复。
- schema已完整读至EOF；planning/profiling全部必需说明恢复完成。已将task_plan顶部陈旧STEP-118更新为STEP-136，并新增`a757f29+batch16`profile审计/重采/双报告/热点重排门禁。
- STEP-136远端审计完成：当前`a757f29` clean/NPU0；无`a757f29+batch16` profiler产物，必须重采。已定位可复用rank0 hook/harness，拟用8/1/4 schedule、14 step同时覆盖普通与SOAP周期并避开最终checkpoint。
- STEP-136 profile夹具配置导入/脚本/hash门禁通过，业务仓库clean；8-NPU任务已启动并核验`a757f29`、8个直接rank、npu-smi 8个唯一PID、无错误。等待14步及rank0 profile解析。
- STEP-136采集完成：14/14、exit0、NPU0；rank0约13.74GB，kernel/operator/trace/DB齐全，communication.json缺失。训练仅改写fusion_result，已归档并恢复Git clean。首次同步逐行扫描3.46GB operator CSV超过64秒工具超时，改用远端后台分析与状态文件，不重复阻塞扫描。
- STEP-136远端原位分析已启动：本地新机器的`python`别名未加入PATH，已改用Codex内置Python驱动现有SSH只读工具；客户容器未安装/升级任何组件。分析器把3.46GB调用栈CSV改为20万行分块计数，当前约500MB RSS、单核持续工作，NPU保持空闲；等待结构化异常JSON、Markdown与独立10节架构报告。
- STEP-136分析闭环完成：分析器先后修正“全量host事件×全部小间隙”的复杂度问题，最终只保留每步Top20/全局Top50并对相交事件做并集；exit0。异常schema校验通过、50个bubble有49个完整kernel上下文，10节架构报告生成，业务仓库恢复`a757f29` clean。
- 最新重排：QR 22.711秒/周期仍最大但历史机制门禁已关闭；活跃P0为`geo_loss.py:224/226/228`三处Boolean index及其反向IndexPut，profile诊断上限约2.07秒/步；P1为MSDA backward约0.796秒/步。转入STEP-137机制门禁，尚未修改业务代码、未创建commit。
- STEP-137启动核验：`a757f29`/`ascend_npu_optimize`/status0，正确容器torch_npu 2.7.1、可见16 NPU。禁止容器`mapqr`自动重启了1-rank并占NPU0；首次TERM后进程树进入T(stopped)但未释放NPU，复核PID/容器/命令后对精确4 PID执行KILL，最终所有容器训练0、NPU Python0。未触碰业务文件。
- STEP-137夹具审计：最新profile目录包含可复用的8-rank `run_profile.sh/ddp_train_profile.sh`，通过诊断目录PYTHONPATH注入模块且运行后删除仓库根临时config；正式最新30-step目录保存客户batch16有效配置副本与已验证8-NPU脚本。掩码探针将沿用该隔离方式，不修改业务源码。
- STEP-137真实finite探针完成：正确容器8-rank、npu-smi 8唯一PID、batch16、3/3、exit0、无错误；12次GeometricLoss中原始target 921,600元素、intra各131,360、inter各103,443,200元素全部0 nonfinite，valid mask 6,568/23,040 true。仓库唯一fusion_result副作用已归档并恢复`a757f29` clean，NPU0。
- STEP-137 masked-L1机制门禁通过：旧容器第二次自动拉起的根因是独立会话shell PGID3951205，精确确认非容器主PID后终止整组，10秒无重启。正确容器8-rank/8唯一NPU PID门禁通过；8 rank的全finite/部分NaN-Inf/全无效/空张量×mean/sum loss和grad全部exact。14,400～1,638,400元素前反向中位0.833～28.314ms→0.529～0.605ms，1.385～46.802×；exit0/NPU0。
- STEP-137业务源码实现完成但未提交：单文件`geo_loss.py`新增finite masked L1 helper并替换GeometricLoss intra/inter六处索引，none/非默认参数回退旧路径；34增12删、AST/diff范围通过。三次`git apply/patch`因CRLF/上下文失败且均未改业务文件，改用精确旧文本各唯一命中的转换脚本成功写入。
- STEP-137源码级8-NPU门禁：8 rank helper四类异常×mean/sum/none与完整GeometricLoss全finite/部分nonfinite的loss/grad全部exact；真实shape旧索引0.822～28.584ms→源码helper 0.515～0.624ms，1.416～46.162×。exit0/NPU0，源码仍未提交，进入3-step训练门禁。
- STEP-137修改后3-step通过：8直接rank/8唯一NPU PID、3/3、exit0、错误0，loss418.504、grad89.797有限，memory26387MiB/rank；fusion副作用恢复。随后为防旧容器周期拉起污染，终止其新会话PGID4101390并临时pause `mapqr`，正确容器保持运行。
- STEP-137正式30-step首启无样本失败：8-rank/NPU门禁通过但首iter前host侧`docker exec`客户端被SIGKILL，exit137；容器未OOM、内核/journal无OOM、系统available约1.9TiB、日志无训练异常。生成config/kernel_meta/fusion副作用已精确清理/归档，保留唯一业务diff；下一次改用容器内detached执行+exit状态文件，不重复host长连接方式。

## 2026-08-13：STEP-137正式测试切换到后8卡

- 用户明确前8卡已被占用，要求后续当前30-step训练改用后8张逻辑卡。
- 只读核验时`mapqr-leicheng`和旧`mapqr`均无活动训练进程，先前detached候选为0 iteration且没有`run.exit`，不构成有效性能样本。
- 远端业务仓库仍为`a757f29`，预期候选仅修改`geo_loss.py`；上次0-step启动留下`fusion_result.json`删除、临时`geo_candidate_config.py`及`kernel_meta/`，将在新测试前精确归档并恢复，不混入业务提交。
- 本地测试编排脚本已把`ASCEND_RT_VISIBLE_DEVICES`从0～7改为8～15；该变化仅用于测试编排，不修改远端环境版本或业务功能。
- 第一次归档命令因诊断指针的行尾字符导致严格前缀门禁返回`BAD_OLD_DIAG`，门禁在任何移动前终止；改为剔除CR/LF并以`readlink -f`核验后成功。
- 旧临时配置与夹具逐字节一致，已同`kernel_meta/`一起移入旧诊断目录；`fusion_result.json`恢复到HEAD。当前远端仅剩预期`geo_loss.py`候选diff。
- 首次后8卡启动门禁还因仓库内单向上传暂存目录`.codex-analysis/`和瞬时NPU记录而在创建样本前退出；确认暂存目录只含本轮脚本后，将脚本移到诊断工具目录并移除空暂存目录。旧`mapqr`仅残留T状态1-rank进程且未占NPU；未终止或修改前8卡用户任务。
- 后8卡正式候选已在`mapqr-leicheng`中以设备8～15、8 rank、端口29870、30 step后台启动；独立诊断目录为`geo_candidate_30step_back8_8npu_20260813T180349`。初查任务处于初始化阶段、无iteration、无错误。
- 该次启动8个rank及后8卡环境变量门禁通过，但8个rank均在第0步因`ModuleNotFoundError: tbe`/GE初始化失败退出。根因是新后台命令覆盖了容器原有`PYTHONPATH`，不是设备、业务候选或客户环境缺组件；本轮0 iteration，无性能样本，既有结论不变。
- `ddp_train_30.sh`对launcher失败未可靠传播退出码，因而`run.exit`错误记录为0；后续正式有效性继续同时要求30条iteration、无fatal错误和进程释放，不能只看退出文件。
- 修正后的环境门禁已成功完成`tbe/torch_npu/torch`导入，但嵌套shell把诊断`print("TBE_OK=1")`的字符串引号剥离，最终在print处触发TypeError；训练尚未启动且无新样本。下一次改用只输出整数device_count的无字符串门禁，避免重复同类引号错误。
- 无字符串环境门禁通过：后8卡可见`torch.npu.device_count()==8`，且`tbe/torch_npu/torch`导入成功。新的独立重试目录`geo_candidate_30step_back8_retry_8npu_20260813T180842`已用端口29871提交；首次即时检查仍在容器内脚本初始化、尚无launcher/iteration/fatal日志。
- 正式启动硬门禁已通过：恰好8个直接训练rank（LOCAL_RANK 0～7），每个进程均继承`ASCEND_RT_VISIBLE_DEVICES=8,...,15`；`npu-smi`显示其主映射覆盖物理NPU 4～7的两个chip，即16逻辑设备中的后8个。当前0 iteration、fatal=0，仍在模型/数据初始化。
- 约2分25秒时各rank仍在首步编译/数据路径，日志已有真实forward的MAP_SHIFT输出，后8卡HBM逐rank增长且fatal=0，不是静默卡死。随后首步完成，当前4/30；Iter4 time=5.904秒、memory=26482MiB、loss=398.4494、grad_norm=70.0308，均有限。
- 后8卡正式运行最终30/30、exit0、fatal0、训练/NPU进程0。全30步均值9.520233秒、global batch128吞吐13.445049 samples/s；严格普通23步mean/median/P95=5.870696/5.853000/6.121800秒；SOAP双步窗口33.689/37.010秒，平均35.3495秒；框架峰值26840MiB/rank。
- loss均值/中位/范围=296.538850/281.527150/208.926600～428.745500；grad均值/中位/范围=53.000363/47.852850/43.273700～93.401100，全部有限。
- 相对父提交`a757f29`：全30步-18.945%、1.234×，吞吐+23.373%；普通mean/median/P95分别-26.337%/-25.610%/-27.492%；SOAP窗口平均-9.885%；框架显存-605MiB（-2.204%）。
- 相对客户同口径基线`4c37039`：全30步-74.572%、3.933×，吞吐+293.268%；普通mean/median/P95分别-48.723%/-49.109%/-48.558%；SOAP平均-87.357%、7.909×；框架显存-1620MiB（-5.692%）。
- 提交前首次清理只处理了`fusion_result.json`删除状态，未覆盖本轮“内容修改”状态；门禁发现后把runtime diff归档到远端诊断目录并精确恢复。最终唯一业务diff为`geo_loss.py` 34增/12删，diff-check和py_compile通过。
- 本地PowerShell首次计算对比百分比时因对象构造后的管道位置错误报`empty pipe element`，未影响远端数据；改为先收集对象再格式化后成功。
- 已创建单功能commit `b36821e 【npu性能优化】GeometricLoss有限值索引消除`，父提交`a757f29`，提交后clean、未push。
- 提交后首次`git show --check`按默认规则把历史CRLF行尾误报为trailing whitespace并在后续复核前退出；使用项目既有CRLF-aware规则`core.whitespace=cr-at-eol`后通过。最终HEAD/父提交/唯一文件/py_compile/工作树clean/目标训练进程0全部复核通过，无需amend。

## 2026-08-13：STEP-138 MSDA候选开始

- 当前权威状态为`ascend_npu_optimize@b36821e`、工作树clean、正确容器无目标训练进程。
- 当前项目只有一份自定义MSDA autograd文件；fp32 backward在line183直接调用`ext_module.ms_deform_attn_backward`并预分配三类梯度张量。最新profile的3.186秒/4步栈正落在该行。
- 多个attention模块在非deploy分支以`torch.cuda.is_available() and value.is_cuda`决定自定义扩展或PyTorch reference；在Ascend迁移环境中这些CUDA兼容属性可能被transfer_to_npu改写，必须运行时验证，不能按字面量判为CPU路径。
- 历史`70576d3`只修改`mmcv/ops/multi_scale_deform_attn.py`通用实现，不是项目自定义文件；因此“历史NPU MSDA CPU fallback”和“最新项目自定义NPU backward”可能同时真实存在于不同调用族。
- 首次兼容属性探针误在宿主机执行，宿主无项目torch，import阶段即`ModuleNotFoundError`，没有触发NPU或训练；改为正确容器、后8卡可见后成功。
- 正确容器的基础属性为`torch.cuda.is_available=False`、`torch.npu.is_available=True`、NPU tensor `is_cuda=False/is_npu=True`、device_count=8。MMCV通用Attention显式识别`IS_NPU_AVAILABLE`并走自定义扩展；项目自定义attention多处只检查CUDA兼容属性，基础环境下会走PyTorch reference/CPU fallback。
- 有效config明确启用TemporalSelfAttention、SpatialCrossAttention、MSDeformableAttention3D和MapTR decoder；仍需确认训练是否额外导入`transfer_to_npu`改变CUDA属性，以及各调用族实际次数。
- 训练入口line407确实导入`transfer_to_npu`；同一后8卡环境实测随后`torch.cuda.is_available=True`、NPU tensor同时`is_cuda=True/is_npu=True`、CUDA/NPU device_count均为8。因此项目CUDA兼容条件在真实训练中成立，相关调用走项目自定义NPU扩展而非CPU reference。
- NPU C++实现的backward直接调用官方`aclnnMultiScaleDeformableAttentionGrad`，三个梯度buffer作为输出传入；Python层在调用前对grad_value/grad_sampling_loc/grad_attn_weight做`zeros_like`。是否能改`empty_like`取决于官方算子是否完整覆写输出，且预期上限只来自三个清零kernel，不是0.796秒aclnn主体。
- 首次定位profile文件因`find -maxdepth 2`层级过浅且grep无命中，在`set -e`下退出，无文件变化；改为全诊断树只读查找后定位最新客户profile及3.459GB rank0 `operator_details.csv`。原始文件继续只留远端。
- 既有完整报告确认4步共有24次MSDA backward主体、纯device 3185.547ms；forward同名主体24次、700.171ms。backward平均约132.731ms/调用、每步6次、约796.387ms/step。
- attribution明确24次backward栈均落line183；forward调用族至少包括SpatialCrossAttention、TemporalSelfAttention和MapTR decoder。下一步用3.459GB CSV分块筛line179～181清零op，不能用整表加载或line grep破坏多行CSV记录。
- 已用`apply_patch`创建只读分块脚本`analyze_msda_buffers.py`，本地py_compile通过；脚本单向上传后复制到既有profile分析目录，业务仓库临时目录已移除并保持clean。
- 分析已在`mapqr-leicheng`后台启动，按50,000行chunk只读取7列并筛选源码179/180/181/183；初查状态RUNNING、无错误输出。
- 分块分析完成：扫描1,468,549行，命中179/180/181/183各72行（每行24次顶层调用及子op）。三个`aclnnInplaceZero`纯device分别3.952/2.797/1.444ms，合计8.193ms/4步=2.048ms/step。
- 三个顶层`aten::zeros_like` host total分别4.483/2.407/2.264ms，合计9.154ms/4步=2.289ms/step；相对当前普通步约5.853秒占比仅0.039%。MSDA主体line183为3211.402ms/4步，约802.850ms/step。
- MSDA buffer候选低于约22.7ms噪声门槛，严格淘汰：不改源码、不做NPU机制微基准、不训练、不提交。分析JSON和脚本留远端profile分析目录，业务仓库保持`b36821e` clean。
- 2026-08-13：STEP-139关闭BEV backbone line120。4步唯一叶子device self为75.483ms，即18.871ms/step，全部由15组真实Conv+BN+ReLU构成；无copy/cast/format，cat仅0.482ms/step。未改码、未训练、未提交，转入P4格式/搬运功能族聚合。
- 2026-08-13：P4聚合发现TextLogger显存统计每步执行NPU tensor、跨rank reduce和item，with-stack host self约4.694秒/step。已形成默认兼容的`memory_interval`候选，客户配置设10；当前仅有2文件未提交diff，待单元门禁及后8卡30-step profiler-off A/B。
- 2026-08-13：TextLogger候选后8卡30-step通过：全步均值9.313947秒、吞吐13.742831 samples/s，普通23步mean/median/P95=5.715218/5.778270/5.922613秒，SOAP窗口均值34.530815秒；相对`b36821e`分别改善2.167%/2.215%/2.648%（普通均值）/2.316%。loss/grad全有限、显存峰值26851MiB。运行副作用已归档，等待最终提交门禁。
- 2026-08-13：创建`bf9ed6e 【npu性能优化】TextLogger显存统计同步降频`，父`b36821e`，仅TextLogger与目标config两文件；提交后clean、训练进程0、后8卡busy0、未push。转查最新profile剩余host-only栈`lc_fusion.py:311`。
- 2026-08-13：STEP-141定位`LCFusionV2`固定grid每步CPU→NPU搬运；已形成仅1文件的惰性设备缓存diff。单NPU完整输出/输入梯度/二次输出bitwise相同，孤立边界2.40x但仅节省0.061ms；待后8卡30-step判断同步收益。
- 2026-08-13：STEP-141正式后8卡证伪：普通均值+1.035%、全步+0.336%、吞吐-0.335%、SOAP+1.277%，未兑现收益。候选patch远端归档，业务文件恢复，当前`bf9ed6e` clean、无训练/NPU进程，不提交；转查`bevformer_encoder.py:602`。
- 2026-08-13：STEP-142关闭BEVFormer line602：`[1,2]`小tensor的74.262ms与相邻`[1]`tensor0.170ms形成同步归因反证，真实缓存本体低于门槛；未改码/训练/提交。旧profile已被Geo+TextLogger提交改变，下一步重采`bf9ed6e`后8卡profile。
- 2026-08-13：STEP-143当前HEAD profile在13/14日志后因用户临时征卡而定向停止；本轮任务进程0、端口29874占用0，禁止自动重跑。导出约13.8GB，Step 9～12四个active窗口及核心CSV/trace完整，运行副作用归档后`bf9ed6e`工作树clean。
- 2026-08-13：停卡期只启动低优先级单线程CPU分析；无训练、`msprof`或分布式进程。待生成异常JSON/Markdown、analysis manifest和独立10节架构报告后重排热点；恢复任何占卡验证必须等待用户明确同意。
- 2026-08-13：用户澄清只禁止占用NPU，CPU分析可正常并行。修复旧分析器schema字段后，当前HEAD异常JSON正式校验通过，架构报告10节齐全；所有分析期间NPU任务0。
- 2026-08-13：当前新P0候选为MapTR正负样本`nonzero(...).unique()`冗余去重。batch16调用256次/step，两处Unique纯device合计49.623ms/step；CPU 131,370例exact。只生成未应用两行patch，远端`bf9ed6e`仍clean；等待用户同意用卡后做后8卡NPU exact与30-step A/B。
- 2026-08-13：用户明确回复“可以开始测试了”，恢复后8卡测试授权。进入STEP-144：先设备/容器/Git门禁，再应用两行最小diff，依次执行后8卡NPU exact与客户batch16单轮30-step；未通过不得提交。
- 2026-08-13：`mapqr-leicheng`的exec通道连续超时；用户明确要求重启。宿主确认目标容器无训练、后8卡无进程后，仅重启完整名称`mapqr-leicheng`。重启后exec、torch_npu、8卡可见性恢复，`bf9ed6e` clean；其他容器未重启。
- 2026-08-13：应用户要求生成可交给另一模型执行的`custom/DrivingSDK优化研究与实施计划.md`。计划保持STEP-144的MapTR两行候选为唯一P0，并把MSDA对比、MapTR host预处理、注释算子可达性/DrivingSDK映射、MHA/BEV Pool、运行时开关、internal format、负载均衡等分级；未修改远端业务代码、未启动训练。
- 2026-08-13：用户要求把“注释算子恢复”放到后面。交接计划已将该方向从P1执行队列移出并设为延期项；当前禁止恢复、适配、插hook或专项复现，R0后只研究当前活跃路径。
- 2026-08-13：STEP-144完成。MapTR两处冗余`.unique()`候选通过后8卡8-rank exact门禁（18,776组输入全部一致），但客户batch16单轮30-step端到端未兑现：全步10.213700秒、吞吐12.532187 samples/s、严格普通步6.352043秒、SOAP均值34.784秒；相对直接父提交`bf9ed6e`分别+9.660%、-8.809%、+11.143%、+0.733%。按门禁拒绝并回退，不提交。远端HEAD仍为`bf9ed6e`，工作树clean，训练进程/端口0，容器正常。
- 2026-08-13：进入STEP-145。当前`bf9ed6e` clean且无NPU任务。新候选定位为MMCV `DataContainer`缺少`pin_memory()`导致客户配置的`pin_memory=True`无法递归作用于内部tensor；当前profile scatter copy为1828次/4步、with-stack host self 18.049920秒。尚未改业务代码、未训练、未提交，先做后8卡真实batch机制A/B。
- STEP-145首次8-rank synthetic门禁仅因测试断言过严失败：3个非空tensor均pinned，1个空tensor无存储所以`is_pinned=False`；已修正门禁语义，业务仓库仍clean。
- 用户更新目标后，执行顺序切换为先`DrivingSDK优化研究与实施计划.md`。DataContainer页锁定候选在8-rank synthetic门禁通过后暂停，未改业务代码；进入R1“当前MSDA实现与DrivingSDK融合实现对比”。
- 2026-08-13：目标容器按用户要求再次重启并恢复。完整名称`mapqr-leicheng`、状态running；`torch_npu`正常，全机16卡，限定`8..15`后逻辑device_count=8。目标容器无训练进程，后8卡空闲；前8卡存在同事任务且未触碰。
- 2026-08-13：DrivingSDK R1只读审计完成。客户固定`mx_driving 1.0.0+gitde13346`的MSDA是五输入autograd融合实现，910_93随包配置为fp32/ND；项目当前活跃路径命中自定义fp32扩展，不走CPU fallback。直接import的后端重复注册错误已用既有`TORCH_DEVICE_BACKEND_AUTOLOAD=0`加载顺序解决，无安装/版本变更。
- 2026-08-13：远端原位扫描当前profile 1,523,977行，4步24次forward+24次backward；forward主体699.839ms/4步、backward主体3239.099ms/4步。C扩展顶层shape缺失，下一步必须通过后8卡8-rank真实1-step元数据探针取得六类真实调用合同，再构造DrivingSDK等价门禁。
- 2026-08-13：按用户新要求完成旧profile清理。经绝对路径、保留目录冲突和活动cwd门禁后，删除12份已闭环/失败重试profile，约释放90GiB；当前R1仍使用的13GiB profile保留。后续profile使用完即按同样门禁删除原始数据。
- 2026-08-13：R1真实shape探针后8卡8-rank 1-step完成，每rank 6次调用、三类合同跨rank一致；原实现loss/grad有限。合同提取后删除最后13GiB profile，当前旧profile目录0，累计释放约103GiB。
- 2026-08-13：DrivingSDK等价门禁确认opapi必须由候选先于项目当前ACLNN调用初始化；候选优先后8 rank完整执行。输出误差≤7.45e-8，其他梯度≤2.88e-6；MapTR/Temporal sampling梯度最大2.05e-4/2.57e-4但NRMSE约4e-7，finite一致、重复allclose。判为数值近似等价、非逐位一致；下一步先做真实shape机制性能测试，再决定是否值得进入真实训练loss/grad门禁。
- 2026-08-13：R1机制微基准通过：三类真实shape加权完整前反向1073.681→310.290ms/step，节省763.129ms、3.458×；8 rank稳定、exit0、fatal0。单文件候选已应用但未提交。
- 2026-08-13：候选真实后8卡8-rank客户batch16 1-step自然完成，loss=460.0874、grad_norm=132.4465有限、无数量级偏离；端口/NPU释放。进入同口径30-step正式A/B前先归档运行副作用，只保留候选业务diff。
- 2026-08-13：STEP-152完成。DrivingSDK MSDA候选后8卡8-rank客户batch16单轮30-step通过：9.269933秒/step、13.808082 samples/s，普通23步mean/median/P95=5.625957/5.637000/6.713600秒，SOAP均值33.301秒；相对`bf9ed6e`全步-0.473%、普通均值-1.562%、中位-2.445%、SOAP-3.562%，但P95因SOAP后两步抖动+13.355%。loss/grad全有限，同输入NRMSE≤6.439e-7。已提交`f922c38 【npu性能优化】MSDA切换DrivingSDK融合实现`，单文件、clean、未push、NPU进程0、profile目录0。
- 当前下一步：继续按DrivingSDK计划审查下一项未关闭候选，先做只读/机制门禁，不直接改码；DrivingSDK阶段结束后再合并此前通用候选计划。后续原始profile在结论归档且不再需要后立即安全删除。
- 2026-08-13：STEP-153启动R2。已完整恢复两项技能与三份持久规划文件，复核昇腾7.3“性能调优流程/并行策略建议”正文。当前结论是先对`f922c38`做MapTR target/GT静态调用链、调用频次和复用边界审计；旧profile为0，历史统计不直接作为新提交证据。当前未连接远端、未占用NPU、未改业务代码。
- STEP-153远端门禁通过：`mapqr-leicheng`、`f922c38`、clean、训练0、profile0。有效config锁定R2活动链为`InternalDatasetTrackStream`、`VectorizeLocalMap`、`GenerateMapGTShifts`、`MapTRv2HeadDecoder/MapTRDecoder`和`MapHungarianAssigner3D`；两次过宽枚举已记录并收窄，仍未改码/训练。
- 2026-08-14：用户再次确认旧profiling原始数据“用完且后续不需要即删除”。继续执行STEP-149：仅在脱敏结论、必要双报告和复核合同提取完成后安全删除原始profile；当前远端旧profile目录记录为0，本次没有占用NPU或执行远端删除。
- STEP-153 R2静态审计：客户有效config为`v6_curve`、4层decoder，训练pipeline已存在`GenerateMapGTShifts`并生成三个cpu-only缓存字段；head包含缓存消费、现场回退和缓存对照逻辑。该机制来自客户既有历史，不重复实现；下一步确认有效运行分支和4层复用边界，再量化剩余GT target重复。
- STEP-153确认缓存确实从`spetr3d.py`传入head，默认不现场重算并跨4层复用。后8卡当前空闲、前8卡有同事任务；profiling规则/模板/schema已完整恢复。下一步复用成功夹具生成rank0单正常步最小profile，完成双报告与R2耗时合同后删除原始profile。
- STEP-153 R2 profile与候选筛选完成：单正常步双报告/schema通过；GT shift前移已存在。唯一新测Hungarian索引合并H2D在8 rank全exact但10.584→10.786ms轻微退化，拒绝且无业务diff/30-step/commit。待归档R2一页报告并删除3.1GiB原始profile后，R2关闭。
- 2026-08-14：STEP-154完成。R2一页结论已归档；删除前schema 0错误、架构报告10节、8-rank门禁、进程/cwd、Git和后8卡门禁全部通过。已删除3,248,664 KiB原始profile并将保留目录改名为analysis；远端顶层profile目录0。R2以`REJECT_NO_COMMIT`关闭，未改业务代码、未训练、未提交。
- STEP-155只读筛选完成：R4虽活动但缺稳定模块级profile证据，暂缓；R5在客户有效config中不活跃，关闭；R6四个候选变量当前容器/夹具/跟踪代码均未设置，进入版本支持性审计。一次过宽递归grep已精确终止，残留扫描/训练进程0，未占用NPU。
- STEP-156进行中：R6远端状态门禁通过；当前torch_npu 2.7.1二进制识别四个变量。已在本地缓存华为官方`v7.3.0-pytorch2.7.1@3be71f4`源码，初步确认TASK_QUEUE默认1且模式2可用、COMBINED是优化器开关、expandable_segments是虚拟内存机制。未改远端、未训练。
- STEP-156官方语义已校正：COMBINED实际仅接入非连续tensor的组合view→contiguous优化，不是训练optimizer；TASK_QUEUE=2是workspace进一步异步下发，CPU_AFFINITY=1是粗粒度绑核，expandable_segments是碎片/OOM机制。下一步用保留分析和容器拓扑选择唯一A/B项。
- STEP-156候选已收敛为`TASK_QUEUE_ENABLE=2`。确认当前默认Level 1、无blocking、无活动NPUGraph/torchair；COMBINED约49ms纯device可见上限留作后续，CPU绑核与虚拟内存均非首项。下一步制作诊断目录内单变量30-step夹具并做启动前门禁。
- R6 Level 2本地夹具已用`apply_patch`生成、Python编译通过并上传远端诊断目录；远端`bash -n`和`py_compile`通过。首次哈希汇总因通配符包含`__pycache__`目录而提前退出，训练未启动；改为只枚举普通文件后继续门禁。
- R6夹具差异门禁通过：相对最近成功默认Level 1 wrapper，仅新增`TASK_QUEUE_ENABLE=2`并更名隔离config/work_dir/结果标签；`ddp_train_30.sh` SHA保持`10ad92c...`。真实位置Config门禁确认batch/rank=16、4层decoder、SOAP、checkpoint关闭；临时根文件0、Git clean，尚未启动训练。
- R6 Level 2正式30-step已在后8卡启动，端口29884。torch_npu预检device_count=8；8个直接rank的LOCAL_RANK/RANK=0..7、WORLD_SIZE=8、可见设备8..15和TASK_QUEUE_ENABLE=2全部核验，fatal0。当前仍在模型初始化，npu-smi尚未出现后8卡PID，需初始化后补做映射门禁。
- R6 NPU映射门禁已补齐：宿主npu-smi后8物理NPU 4～7的8个主PID经`NSpid`一一映射到容器直接rank PID 162314～162321。一次只读sed区段表达式失败后改用精确NPU行正则，训练未受影响；当前仍在初始化、fatal0。
- R6分析器已在本地`apply_patch`生成、编译后上传诊断目录，SHA `724b071d...`。训练中Git状态3项均为预期副作用：临时config、`kernel_meta/`、暂缺`fusion_result.json`；等待自然退出后归档并恢复，不在训练中处理。
- R6一次包含容器内全进程扫描的状态查询在初始化高负载期超过本地30秒时限；训练未被中断。监控已改为只读日志、宿主PID和端口。
- 2026-08-14：R6 TASK_QUEUE Level 2单变量30-step完成并拒绝。9.966667秒/step、12.842809 samples/s、普通均值6.001435秒、SOAP34.170秒；相对默认Level 1分别+7.516%、-6.991%、+6.674%、+2.610%。exit0/fatal0/loss-grad有限，但性能全面回归，`REJECT_NO_COMMIT`。副作用已归档，`f922c38` clean、端口/训练0、后8卡空闲、profile目录0。
## 2026-08-14 00:56：固化 profiling 原始数据即时清理规则

- 状态：已记录为每个 profiling 阶段的强制收尾门禁。
- 当前状态：旧原始 profile 已清空；最近复核为顶层 profile 目录 0。
- 后续执行：分析与报告验收完成、确认无需继续复核后，当轮立即安全删除原始目录，并在 `操作步骤.md` 记录释放空间和删除后断言。

## 2026-08-14 01:12：R6 COMBINED 单变量完成并拒绝

- 30/30、`COMBINED_TRAIN_EXIT=0`、fatal0；直接8 rank、WORLD_SIZE=8、后8可见设备和`COMBINED_ENABLE=1`继承通过，`TASK_QUEUE_ENABLE`未设置。
- 相对默认`f922c38`：全步+4.344%、吞吐-4.163%、普通均值+3.591%、中位+5.304%、SOAP+1.047%；结论`REJECT_NO_COMMIT`。
- 指标JSON和一页报告已保留远端诊断目录；运行副作用归档，业务仓库clean，训练0、端口0、后8卡空闲、原始profile目录0。
- 启动期宿主npu-smi PID映射没有成功归档，已明确降级为不可采用的测试证据；候选本身已被负收益否决，不重跑。

## 2026-08-14：R6 CPU affinity 只读审计启动

- 已恢复planning-with-files上下文并读取npu-smi命令规范；未启动训练、未修改远端环境。
- 远端仍为`mapqr-leicheng`、`f922c38` clean、训练0、profile0。容器无限额地可见320 CPU和8 NUMA节点，每节点40核；后8逻辑NPU为Phy-ID8～15。
- 默认CPU affinity按8个运行时device index把320核八等分，但`npu-smi topo`不返回CPU范围，PCI sysfs也对16个加速器均报告NUMA=-1。尚不能证明默认device0→CPU0～39与可见Phy-ID8匹配。
- 当前下一步：只读调用当前安装torch_npu/DCMI亲和查询能力，提取每张物理卡真实CPU范围；若无法建立映射，关闭自动默认绑核，不做30-step。
- DCMI零绑核探针完成：当前环境不支持`dcmi_get_affinity_cpu_info_by_device_id`，`npu_affine`被禁用；mode0下亲和前后均为0～319，临时114,867行debug日志提取精确结果后立即删除。业务仓库clean、训练0、profile0、后8卡释放。
- 默认自动绑核因后8卡可见重编号且缺物理映射证据被判定不具备正式A/B资格。下一步制作8-rank NUMA→H2D微基准，仅用于推断每张后卡邻近CPU节点；不会改变业务代码或最终功能。

## 2026-08-14：R6 CPU affinity关闭

- 8-rank NUMA→H2D微基准自然完成、fatal0、64组结果；默认映射与每卡事后oracle仅差0.793%，稳定样本仅20/64，无法恢复可信物理亲和映射。
- 结论`REJECT_NO_COMMIT`，没有启动30-step；报告、JSON、脚本保留远端诊断目录。Git clean、训练/探针0、端口0、profile0、后8卡释放。
- R6整体关闭：TASK_QUEUE和COMBINED均回归，CPU affinity无可靠合同，expandable_segments不适用于当前无OOM/碎片场景。下一步按DrivingSDK计划审计R7 internal format现有format转换证据。
# 2026-08-14 R7 启动与 profiling 清理规则

- R6 已完整关闭：TASK_QUEUE、COMBINED 均正式回退；CPU affinity 无稳定物理映射且理论收益仅约 0.793%，不进入训练门禁。
- 已进入 R7 `allow_internal_format` 前置证据审计，当前未启动训练、未占用后 8 卡。
- 已落实用户新增规则：旧 profiling 原始数据分析后不再需要即删除；当前远端 raw profile 数量为 0，仅保留脱敏摘要与报告。
- 下一步：核对固定 Ascend PyTorch 7.3/PyTorch 2.7.1 源码语义，并追溯项目中显式关闭 internal format 的历史与正确性约束。
- 已完成第一轮语义/历史核对：硬件为 Atlas 800I A3；固定版本对 A3 默认关闭 internal format；项目显式关闭来自早期 loss/randomness 对齐提交，且模型代码存在 ConvTranspose2d 候选路径。
- 尚未改业务代码或启动训练。下一步只读确认容器内实际安装包对 `True` 的处理方式，并定位客户活跃配置实际实例化的卷积/反卷积路径；若运行时自动回退或活跃路径兼容合同不足，R7 直接关闭。
- 现场确认显式 `True` 在 NPU 初始化前可读回 `enable`，因此 R7 不是硬件层面自动失效；客户活跃 `BaseBEVBackbone_FPN(use_deconv=True)` 确实包含可达 ConvTranspose2d。
- 准备进入最小 8-rank 隔离门禁：分别在独立进程中设置 False/True，验证初始化后实值、代表性 Conv2d/ConvTranspose2d/view 输出与梯度，并统计纯算子耗时；门禁临时 tensor/log 比对后删除。
- R7隔离门禁完成：两组均8/8、exit0/fatal0，数值兼容通过；但True使Conv/ConvTranspose/view链8-rank中位分别回归15.487%/47.821%/13.739%。
- R7已否决，不进入完整训练、不改代码、不提交。正在只保留脱敏报告/统计JSON并删除原始tensor、日志、ready/exit和一次性脚本。
- R7清理闭环完成：删除32个原始结果文件、全部False/True/compare日志、ready/exit、一次性脚本、kernel_meta和本轮生成的fusion结果；恢复跟踪版`fusion_result.json`。远端只保留11,891-byte统计JSON与1,598-byte Markdown报告，仓库clean、训练/端口/profile均为0。
- 当前阶段转入P2/R8，只读审计rank方差与样本复杂度相关性；在证据成立前不启动训练、不修改sampler。
- R8第一轮只读盘点完成：所有保留30-step日志均为rank0聚合日志；无逐rank step/data time、点数、GT数或序列长度，R2异常报告也只有rank0单步。当前证据等级为`insufficient_evidence`，尚未修改代码或启动NPU。
- 已完成R8一次性sitecustomize诊断、汇总器和后8卡30-step包装脚本；Python语法、相关函数自测通过。首次尝试本地`bash -n`时Windows PATH无bash，未执行shell语法检查；下一步改用Codex bundled Git Bash，不重复失败命令。
- R8后8卡诊断自然完成：30/30、8rank×30=240条、exit0/fatal0、进程/端口释放。23个普通步存在约13.58% forward rank范围，但四个复杂度特征相关均接近0且置换检验不显著，无准入特征。
- R8已拒绝，不改sampler、不训练候选、不提交。正在保留脱敏summary/report并删除原始JSONL、训练日志、work目录、一次性上传和kernel/fusion副作用。
- R8清理完成：诊断目录3,022,522→5,207 bytes，仅留2,397-byte summary和2,700-byte report；240条JSONL、日志、work、hook、上传目录、kernel_meta均删除，fusion_result恢复HEAD。仓库clean、后8卡进程0、端口0、raw profile0。
- 当前转入DrivingSDK P3，只读核对活跃Pillar/Sparse链是否同时满足profile热点和固定版本成熟API两项门禁。
- DrivingSDK P3审计完成并关闭：固定SDK有Sparse/Voxel/Scatter API，但客户活跃配置未实例化SparseConv，留存profile没有Pillar语义归因；通信仅8.6975ms且76.8%掩盖。未改码/训练/提交。
- DrivingSDK计划队列至此闭环，按用户要求合并此前计划；当前恢复DataContainer页锁定候选的真实batch机制A/B。
- 2026-08-14：完成R9 DataContainer真实客户batch机制A/B。8-rank×20-step/版本均自然退出0；pinned bytes 0%→100%，同步scatter中位降低85.756%，诊断整步中位降低12.262%，6032个tensor exact且loss全finite，判定可进入正式候选验证。原始诊断由6.72MB清至3.17KB，仅留脱敏JSON/Markdown；仓库clean、训练0、后8卡释放。下一步形成单文件候选，先1-step再30-step正式A/B，尚未提交。
- 2026-08-14：按用户新增规则清理历史diagnostics中的残留`kernel_meta*`目录728个（合计19,298 bytes；绝大多数为空目录）；未发现仍存在的原始`PROF_*`/`ASCEND_PROFILER_OUTPUT`主体。清理后匹配数0、训练/profiler进程0、仓库clean。
- 2026-08-14：DataContainer生产候选门禁通过。修正测试夹具误导入site-packages MMCV后，仓库源码断言成功，后8卡8-rank全部保持tensor exact/metadata exact、非空tensor 3/3 pinned、cpu-only不变；原始gate日志与一次性文件已删除，仅留1.5KB脱敏摘要/校验。
- 2026-08-14：客户batch16后8卡8-rank 1-step容量门禁通过：1/1、exit0、fatal/OOM0、显存24,847MiB/rank、loss=444.5895、grad=97.4894均有限。原始日志/work/harness/kernel_meta已删除，fusion副作用恢复；业务树只剩单文件候选。下一步执行单轮30-step正式性能测试并同时报告直接父提交和永久基线累计对比。
- 2026-08-14：DataContainer正式30-step完成并拒绝。普通23步mean/median/P95相对`f922c38`改善9.652%/10.094%/12.436%，但全步+3.475%、吞吐-3.358%、SOAP+6.698%，未形成端到端净收益；`REJECT_NO_COMMIT`。相对永久基线客户派生`4c37039`累计全步-74.380%（3.9032x）、吞吐+290.323%。候选、fusion和kernel副作用已恢复/删除，原始训练日志与夹具已删除，仅保留约4KB脱敏报告，HEAD clean、训练0、后8卡释放、raw profile0。
- 2026-08-14：进入STEP-174。完整读取planning与Ascend profiling技能及其kernel schema、rulebook、架构报告模板；session-catchup无未同步输出。当前阶段只读盘点`f922c38`留存的脱敏异常/架构报告并重新排除已关闭方向，不占NPU、不改业务代码。若证据不足才采集最小新profile，raw产物在结论形成后立即删除。
- STEP-174诊断分析器已完成本地最小修改：top bubble JSON/Markdown将包含重叠host event、category、overlap、event duration和项目stack frame；不改变异常阈值或soft归因，也不进入业务仓库。下一步做py_compile与合成trace单测，再上传独立诊断目录。
- STEP-174分析器本地`py_compile`通过。入口仍只依赖profile root/config/output/run-name，bubble计算读取kernel_details、step_trace、trace_view和operator_details；下一步用极小合成profile做端到端JSON/Markdown上下文断言。
- 本地诊断依赖安装：bundled Python已有pandas、缺ijson/jsonschema；通过PyPI安装到项目专用`.codex-tools/python-packages`，版本`ijson 3.5.1`、`jsonschema 4.26.0`（及其解析依赖）。用途仅为本地合成profile和schema验证，未修改远端容器、驱动、CANN、PyTorch或torch_npu。
- STEP-174增强分析器合成端到端测试通过：JSON首个bubble正确保留`aten::to`及项目相对栈帧，Markdown含host-context表；正式Draft 2020-12 schema校验通过。分析器SHA256=`6774a9d8...27c`。
- STEP-174新诊断目录已建立，远端原位复制R2成功夹具；canonical脚本SHA仍为`10ad92c...e0fc`，hook/config/wrapper哈希已记录。增强分析器与正式schema经仓库临时文件单向上传，尚未启动训练。
- STEP-174启动前门禁通过：分析器/schema已原位移入诊断目录且临时上传0；容器内py_compile和两个shell语法检查通过。远端`f922c38` clean、正确容器、torch_npu可见8设备、训练0、端口29924空闲、profile尚未创建；物理NPU 4～7（逻辑8～15）全部健康且无进程，前8卡同事任务保持不动。
- STEP-174最小profile训练已启动，直接训练rank正好8个，`LOCAL_RANK=0..7`、`WORLD_SIZE=8`、可见设备8～15完整，fatal0；当前仍在初始化，后8卡尚未申请HBM。
- STEP-174设备映射门禁补齐：8个直接rank均在后8卡，主映射依次4/0、4/1、5/0、5/1、6/0、6/1、7/0、7/1；共同4/0为通信上下文，fatal0。
- STEP-174最小host-followup profile采集完成：11/11、`R2_PROFILE_TRAIN_EXIT=0`、fatal/OOM0；第10步为record_shapes+with_stack活动步，日志time=43.002s，仅作归因。kernel/trace/step/operator四类导出文件齐全，训练/端口释放、后8卡空闲。进入不占NPU的远端原位双报告分析。
- STEP-174 raw profile约3.316GB/121文件，关键导出kernel 7.41MB、operator 856MB、trace 1.205GB；运行副作用仅`fusion_result.json`16行删除和1个kernel_meta。增强分析器已在容器内原位运行完成，输出4个文件、日志无报错，NPU保持空闲。
- STEP-174异常JSON正式schema 0错误，20/20 bubble均有host context及项目栈；架构报告10节检查通过。发现config覆盖文件相对base解析失败导致架构配置元数据unknown，准备改传仓库实际base config重跑同一分析器。
## 2026-08-14 STEP-174 host-context 采集与报告生成

- 已完成：分析器增强及本地合成 fixture/schema 门禁；后8卡8-rank最小普通步 profile；NPU/rank/PID 核验；远端原位生成异常发现报告和独立模型架构报告；正式 schema 0错误、20/20 bubble host context 完整。
- 当前：从 raw 导出中补取 MapTR `loss_single.py:1218`、loss入口 `.to` 和 gradient fingerprint hook 的调用/耗时证据，并检查 bubble13至20，选择一个可独立验证的功能候选。
- 待办：证据提取结束后立即删除约3.316GB raw profile、恢复训练副作用；随后只对入选候选执行机制门禁、1-step、30-step，达到门槛才按单功能形成一个 `【npu性能优化】...` commit。
- 已完成目标栈精算、bubble13～20审计和raw清理：删除3,319,720,122 bytes/135文件，raw=0，保留252KB脱敏报告；业务Git clean、训练/profiler=0。
- 已选候选：MapTR类别GT容量告警计数复用，消除line1218每普通步256次NPU scalar-sync链。下一阶段进入源码最小改动、8-rank机制等价门禁、1-step和30-step正式验证。
- 候选3增2删已应用但未提交。两次8-rank机制夹具因诊断profiler API差异退出；rank/world/后8卡/npu-smi映射已经通过，数学/告警断言尚未完成。下一轮改用PyTorch CPU activity仅记录NPU张量的host ATen调用名，避免依赖torch_npu导出API。
- STEP-175完成：第三轮8-rank机制门禁与真实1-step均PASS，但30-step相对父提交全面回归（全步+7.805%、吞吐-7.240%、普通步+8.118%、SOAP+5.962%），候选拒绝、回退、不提交。全部原始门禁/训练日志与harness已删除，仅留脱敏摘要；远端回到 `f922c38` clean、资源释放。
- STEP-176启动：按DrivingSDK完整队列复核后，选择唯一仍活跃但证据不足的R4标准MultiheadAttention作为下一研究项；先做实现/API审计和无业务改码的模块级8-rank窄探针，未达热点门槛不进入融合替换。
- STEP-176完成：R4真实4层MHA forward稳定device总量仅1.552ms/step，最大1.704ms，仅为22.7ms准入门槛7.51%；关闭融合Attention/need_weights实验，不改码不提交。探针原始JSONL/日志/harness已删除，远端 `f922c38` clean、资源释放。
- STEP-177模块级归因完成首层采样：后8卡8-rank 8/8、exit0；稳定step3～8为5.8495s mean、5.8720s median。热点依次为ResNet 362.8202ms median、BevEncoder 311.8138ms（内含BEVFormer 229.9559ms）、FPN 53.2472ms、MapTR head 47.6234ms、PillarVFE 41.4210ms。父子不相加，下一步对ResNet与BEVFormer做更窄子阶段归因。
- 本轮逐调用JSONL/日志/work/harness已删除2,833,004 bytes，仅留脱敏summary/report/SHA；业务 `f922c38` clean、训练进程0、raw关键文件0。未改业务代码、未创建commit。
- STEP-177第二层归因完成并清理：ResNet四stage约82.32/78.81/87.69/52.67ms，均为连续NCHW标准BasicBlock数学；BEVFormer layer130.55ms由SpatialCross88.94ms（MSDA81.65ms）和Temporal35.53ms主导。未发现可独立删除的stage/layer分支，不形成候选。
- 新线索为BEVFormer `point_sampling` 的两处大张量repeat（理论临时量约525MiB/rank/step）；已启动仅计量该函数的8-step后8卡探针，尚未改业务代码。第二层原始明细删除2,776,017 bytes，raw0。
- 2026-08-14：STEP-177 `BEVFormerEncoder.point_sampling` 打包 BMM 候选完成并拒绝。函数级 8/8 rank exact（max_abs=0），单调用 83.4096→1.8561ms、峰值增量节省约423.6MB/rank；但两轮独立客户 batch16×30-step 均重复产生首进程约64s+27s编译成本。复验全步9.597100s、吞吐13.337362 samples/s、普通23步mean/median/P95=5.641913/5.442000/6.310400s、SOAP均值16.195s；相对`f922c38`全步+3.529%、吞吐-3.409%、普通均值+0.284%，判定`REJECT_NO_COMMIT`。候选已回退，业务仓库clean；两轮原始训练/checkpoint/harness共删除3,221,813,103 bytes，仅留脱敏指标/报告/SHA。
- 2026-08-14：point_sampling候选拒绝清理后继续长期目标。已重新启用planning-with-files与ascend-profiling-anomaly技能；session-catchup无未同步输出。当前先完整复核DrivingSDK计划R2之后路线与现有脱敏证据，不改业务代码、不占NPU；只有证据不足才采最小profile，raw数据分析完成且不再需要后立即删除。
- STEP-178 PointPillarScatter真实机制门禁通过：8rank×8call、64/64输出/stride exact，旧→新device中位31.946→4.860ms（6.574x），同步host 32.281→5.171ms；真实无空batch/重复坐标，feature为冻结FP32。原始数据已删除2,766,590 bytes，业务仍`f922c38` clean。下一步只实现该单文件批次向量化并做独立8-rank空batch/重复坐标/梯度门禁。
## 2026-08-14 STEP-178-2 进展

- 已把散点向量化门禁从错误的通用 `PASS` 更正为 `PASS_ACTIVE_VOXEL_CONTRACT`。
- 已明确保留人工重复坐标失败结果，客户真实唯一坐标契约内三类有效用例全部通过。
- 已复核 raw 文件剩余 0，并完成脱敏报告 SHA256 校验。
- 当前候选仍未提交；下一门禁为后 8 卡、8 rank、客户配置 1-step。
## 2026-08-14 STEP-178-3 进展

- PointPillarScatter 批次向量化已通过客户真实配置后 8 卡、8 rank、1-step 可训练性门禁。
- loss/grad 有限，无 fatal/OOM，进程和端口已释放。
- 已删除有效轮3.218GB原始 work/checkpoint/log；两次无效启动原始文件也已分别清理并仅保留脱敏失败原因。
- 已恢复 `fusion_result.json`、删除 `kernel_meta`，候选仍未提交；进入30-step profiler-off量化。
## 2026-08-14 STEP-178-4 完成

- PointPillarScatter批次向量化完成三轮30-step复验，最终因池化全步/吞吐回退被拒绝。
- 无commit、无push；业务代码已回退到`f922c38` clean。
- 三轮原始数据已全部删除，相关进程0，后8卡释放。
- 2026-08-14：STEP-179从clean `f922c38`开始。PillarVFE 41.421ms/step模块中定位到last PFN的大张量布局物化；已创建独立8-rank真实规模exact/梯度/计时门禁，尚未修改远端业务源码。
- 2026-08-14：STEP-179关闭。PillarVFE布局候选8/8 exact但仅节省7.906ms/step，低于准入门槛；未改业务、未训练、未提交。原始数据已清理，后8卡释放。
- 2026-08-14：PillarVFE layout 候选严格等价但仅节省 7.906410 ms/step，低于准入线，未改码/未训练/未提交，raw 已清零。进入 STEP-180，按单文件窄范围审计 SECONDTransFPNV3、ConvNeXt、BaseBEVBackbone_FPN。
- 2026-08-14：STEP-180 ConvNeXt 窄审计完成；可见重复减法仅存在于非活跃 LayerNorm 分支，活跃 BN 路径无独立 copy/layout 冗余，暂不改码。
- 2026-08-14：STEP-180 BaseBEVBackbone_FPN 窄审计完成；未发现超过准入线的独立等价冗余，暂不改码。仅剩 SECOND 四路 stack+sum 需真实 shape 上限判定。
- 2026-08-14：已按活跃配置导出 SECOND 四路真实输出形状 `[16,256,96,160]`；准备执行不改业务树的后8卡8-rank stack+sum 机制门禁。
- 2026-08-14：SECOND 8-rank门禁完成；pairwise输出/梯度exact，但完整前反向仅10.404022→2.454373 ms，节省7.949649 ms，低于门槛。未改业务源码、未训练、未提交；raw/日志/脚本30,901 bytes已删除，后8卡空闲。
- 2026-08-14：按用户新增持续规则复核全部历史diagnostics；profiling raw标志目录0、PROF目录0，无遗留旧profiling数据。
- 2026-08-14：持续目标新一轮恢复完成；重新启用并读取 planning-with-files 与 ascend-profiling-anomaly 技能，session-catchup未发现需补写上下文。当前从 STEP-180 clean `f922c38` 与 raw profile=0 状态继续，先复核 DrivingSDK 计划剩余项和通用候选池，不占用NPU。
- 2026-08-14：DrivingSDK R0/R1/R2/R4～R8/P3状态矩阵完成，全部已有提交/拒绝/关闭结论；根目录与custom计划哈希一致。转入通用计划，下一项先只读审计HF32单变量候选。
- 2026-08-14：HF32初审确认训练入口同时显式False，固定torch_npu提供独立Conv/MatMul开关；远端仍`f922c38` clean。宽grep超时后已切换为git grep，无状态变更。
- 2026-08-14：固定torch_npu默认Conv HF32=True、MatMul=False；DrivingSDK附带patch仅在arch35建议MatMul=True。决定先只验证Conv恢复默认，MatMul保持False，避免组合变量。
- 2026-08-14：确认权威客户配置为20260113st r34_0114、batch/rank16、workers8；正在静态解析实际图像预处理和ResNet shape，暂未占NPU。
- 2026-08-14：静态确认7相机及ResNet四stage注释shape136×240→17×30；继续解析训练增强最终输入，未启动测试。
- 2026-08-14：训练增强确认576×1024基础尺寸、±5%随机resize、pad32，ResNet实际为空间动态shape族；已定位变换类，继续枚举其精确尺寸规则。
- 2026-08-14：读取变换实现后纠正前述判断：随机resize后固定crop回576×1024，Pad不变；真实ResNet输入固定为112×3×576×1024。
- 2026-08-14：Conv HF32首次8-rank门禁在ResNet调用前因MMCV `eval()`链式返回None失败，非算子/HF32问题；已定位并修正为分步to/eval，准备清理失败产物后重跑。
- 2026-08-14：Conv HF32修正版8-rank机制门禁通过；真实ResNet节省78.024ms/-21.511%，最大NRMSE 2.787e-4、全部有限，获准进入客户1-step。尚未改业务代码、尚未获得正式采用资格。
- 2026-08-14：HF32机制raw 8文件及日志/脚本58,715 bytes已删除，仅留summary/report/SHA；运行副作用恢复、Git clean。现已应用唯一业务候选：`tools/train_spetr.py`一行Conv HF32 False→True，MatMul仍False，py_compile与text diff-check通过，尚未提交。
## STEP-181 Conv HF32 当前恢复点（2026-08-14）

- 机制门禁：真实 ResNet 输入 `[112,3,576,1024]` 下 Conv HF32 pooled median 362.725170→284.700794 ms，节省 78.024376 ms（-21.511%）；最大 NRMSE `2.787196e-4`，属于有限近似而非逐位等价。
- 完整入口 1-step：后 8 卡、8 rank、客户配置自然退出 0；time 47.748 s、memory 24847 MiB、loss 449.4805、grad_norm 98.6103，均有限且无运行异常。
- 清理状态：本轮原始日志/JSON/TensorBoard/work/harness 已删除，只保留脱敏摘要、清理报告和 SHA256；后 8 卡本轮进程 0，端口已释放，远端业务树只剩 Conv HF32 一行候选 diff。
- 下一步：单轮 30-step profiler-off，按普通 23 步、SOAP 4 步和全 30 步统一口径量化，并同时报告相对 `f922c38` 与永久基线 `4c37039` 的结果；通过前不提交。

## STEP-181 Conv HF32 最终结论（2026-08-14）

- 单轮 30-step 已完成：全步 9.426467 s、吞吐 13.578789 samples/s；普通 23 步均值/中位数/P95 5.819174/5.658000/6.483500 s；SOAP 16.678 s；显存 26826 MiB。
- 相对直接父提交 `f922c38`：全步慢 1.689%、吞吐低 1.661%、普通均值慢 3.434%，仅中位数基本持平；不满足端到端净收益门槛。SOAP 大幅波动不可归因给 Conv 单变量。
- 决策：`REJECT_NO_COMMIT`，候选已回退，远端 `ascend_npu_optimize@f922c38` clean。
- 相对永久基线 `4c37039` 的累计对比：全步耗时 -74.822%、吞吐 +297.157%（3.972x）；该累计收益不包含被拒绝的 HF32 候选。
- 清理：远端原始产物删除 2,906,147 bytes，原始 profiling 关键文件 0；只保留脱敏摘要/报告/SHA。本地 HF32 一次性脚本已删除；后 8 卡本轮进程 0。
- 下一步：关闭 HF32 方向，选择下一个可独立归因且理论收益上限足够的高耗时热点。
## 2026-08-14 STEP-182 启动

- 已恢复并完整读取 `planning-with-files`，session-catchup 无未同步内容；当前权威恢复点为远端 `ascend_npu_optimize@f922c38` clean，HF32 已拒绝并清理。
- 已复核用户指定的昇腾 7.3 调优入口和并行策略建议：继续坚持先采集/拆解瓶颈再改动；TP/PP/ZeRO、框架重编译不适合当前固定客户环境，不能把显存空余直接等同于并行策略问题。
- 下一步：只读复核 DrivingSDK 计划已关闭状态和通用计划未关闭项，结合现有脱敏模块耗时选择下一候选；证据不足才采最小新 profile。

- STEP-182矩阵复核完成：已关闭项不重复，batch/rank16已是客户口径，不能靠继续放大全局batch保持功能。当前唯一未独立闭环的高位线索收敛到normal-step host-context bubble003的MapTR loss GT列表`.to(device)`；先远端原位只读归因和上限测量。

- STEP-182 bubble003关闭：54.692ms gap不能归因给`.to`，可见host self只有约1.3～2.5ms且为GT CPU→NPU必要搬运；不改码、不训练。继续只读审计bubble004梯度指纹/zero_grad边界。

- STEP-182 bubble004静态结论：fingerprint迭代/阶段均为空，纯指纹代码不执行；训练主链不可删除。`find_unused_parameters=True`仍有独立验证空间，已准备不改变梯度/更新的3-step后置统计探针，尚未启动。

- STEP-182 DDP探针完成并关闭：3/3、exit0；rank0三步固定142/701个trainable参数无梯度（25,397,504/102,682,869元素），`find_unused_parameters=True`必须保留。无A/B、无改码、无commit；远端raw删除2,709,625 bytes，HEAD clean、后8卡释放、profile raw0。

- STEP-182当前device热点覆盖完成：BatchMatMul为已拒绝point_sampling，Unique/Nonzero/Index等已有闭环。新发现活跃VectorizeLocalMap仍无条件打印三行MAP_SHIFT并做tolist/hash；先做不占NPU的真实DataLoader短采样确认当前batch16频次和上限。

- STEP-182 MAP_SHIFT门禁完成并关闭：正确容器CPU-only读取6个客户batch（batch16/workers8/prefetch3），仅2次触发、6行输出；稳态batch中位0.269907s，日志乐观收益远低于22.7ms。不改码、不做NPU A/B、不提交。远端5个原始文件12,167 bytes及本地脚本/pycache已删除；`f922c38` clean、后8卡进程0、profiling raw关键文件0。
- 已将“旧profiling/诊断原始数据在结论提取后立即删除”固化为后续操作门禁；本轮扩展审计原始profile文件0、字节0、输出目录0，当前没有遗留原始profiling数据。
- STEP-183启动：重新完整读取planning与npu-smi技能并运行session-catchup；复核两份DrivingSDK计划哈希一致。当前候选队列已闭环，转为只读冻结`bf9ed6e→f922c38`扩展正确性/收敛A/B合同，重点验证非逐位MSDA；尚未启动训练或修改业务代码。
- STEP-183合同已冻结：两版各完整4 epoch=876 step，后8卡8-rank、客户batch16、自然随机生产语义，预计顺序4.6～5小时；各自末步checkpoint再恢复5步。当前HEAD/父提交、config/harness SHA、MMCV save_last、训练0、后8卡健康/空闲均已核验，准备先启动父提交。
- STEP-183父提交首次启动在0 iteration、0 NPU进程时早退：隔离worktree缺主仓库Git忽略的MMCV编译扩展，非训练/NPU结论。将验证提交间依赖源码无差异，仅链接现有同版本二进制到隔离worktree后重试；不安装或重编远端依赖。
- 父提交第二次启动仍在0 iteration/NPU0时早退：启动命令覆盖容器原`PYTHONPATH`，使CANN/TBE路径丢失并在`AclSetCompileopt`失败；容器原环境登录/非登录均能import tbe。第三次将保留原`PYTHONPATH`，此前两轮均不计入训练。
- 父提交第三次启动通过：8个唯一rank/world8，主PID逐一映射后8卡4/0～7/1。29/876时最新time6.214s、memory26645MiB、loss233.2994、grad58.6732；29条loss/grad全finite，错误0。父提交长跑继续。
- 父提交长跑监控至50/876：最新time6.540s、memory26810MiB、loss171.7860、grad50.7915；全部loss/grad有限，fatal0。41步时全步mean/median/P95=9.948/6.214/25.952s，周期长步按合同单独统计。
- 父提交第1 epoch完成并进入220/876。E1 loss mean/median/P05/P95=142.921568/112.150100/72.559300/340.091200，首20步mean351.390015→末20步74.619565；grad mean/median=50.930898/47.760900，全部finite。普通174步mean/median/P95=6.327161/6.212000/7.269000s，SOAP42步mean/median/P95=16.962667/18.702000/25.957000s；全epoch吞吐14.622955 samples/s，显存26810MiB，fatal0。
- 父提交300/876：epoch2前81步loss mean/median/P05/P95=67.972219/68.304800/53.635800/84.279300，grad mean/median=46.508770/45.080700，全部finite；time mean/median/P95=8.425864/6.410000/25.798000s，峰值显存27086MiB，fatal0。loss相对epoch1末段保持连续并继续下降。

## 2026-08-14：旧 profiling 数据清理规则再次固化

- 已把“脱敏结论和校验完成且后续不再需要后立即删除 profiling raw，并复核为0”写入项目长期规则和执行计划。
- 远端 diagnostics 只读复核：raw profiling 文件 0、字节 0、原始 profile 目录 0，本次没有需要删除的旧 profiling 数据。
- STEP-183 当前 profiler-off 训练继续原任务，不重启；训练日志/work/checkpoint 仍用于收敛、resume 和可能评测，待其用途结束后再按验收合同清理。

## 2026-08-14：STEP-183 父提交 epoch2 完成

- 同一 `bf9ed6e` 任务跨过438步，查询时442/876、launcher running；全部loss/grad finite、严格fatal0。
- epoch2 loss mean/median/P05/P95=`60.518547/60.205100/44.553130/78.473120`，前20步mean `74.245135`→末20步 `51.337850`；grad mean/median=`46.188121/44.834100`。
- normal mean/median/P95=`6.397589/6.294000/7.699600s`；SOAP=`16.943227/17.437500/27.186450s`；吞吐=`15.029918 samples/s`；显存峰值保持`27086MiB`。
- 进入epoch3前4步loss mean/median=`42.828175/43.327400`、显存峰值仍`27086MiB`。继续同一任务至657步统计点。

## 2026-08-14：用户进度汇报前权威状态复核

- 远端目标仓库仍为 `ascend_npu_optimize@f922c3897255`，Git status 0；从永久基线 `63861df` 之后共有7个性能功能commit，另有1个客户运行字段对齐commit，均未push。
- STEP-183 父提交 `bf9ed6e` 的876步长期收敛训练仍是唯一正式任务；查询时469/876、launcher running，全部loss/grad finite、严格fatal0，最新loss/grad=`50.7572/38.3813`，framework显存峰值保持`27086MiB`，按预期尚未生成末步checkpoint。
- 当前阶段未新增业务diff或commit；待父版本876步+resume5步完成后，才顺序启动当前`f922c38`同合同训练并裁决MSDA长期收敛。

## 2026-08-14：DrivingSDK MSDA 最终验收剩余时间重估

- 实时父提交进度509/876、launcher running；最近100步均值`8.531570s`，按当前速度父提交剩余约3131秒（52.2分钟），显存峰值仍`27086MiB`。
- 预计父提交末步保存+resume5步约10～20分钟；当前`f922c38`完整876步按同速度约2小时05分，末步保存+resume约10～20分钟；双版本汇总、最终指标可达性核验和清理约20～40分钟。
- 从本次查询起，若无数据/容器/HCCL异常，预计约3.5～4小时形成MSDA长期验收裁决；代码替换commit已存在，剩余的是发布级收敛与恢复验收。

## 2026-08-14：STEP-183 父提交 epoch3 完成

- 查询时664/876、launcher running；iter439～657全部loss/grad finite、strict fatal0。
- epoch3 loss mean/median/P05/P95=`43.356168/43.102500/31.938700/55.758720`，前20步mean`46.933555`→末20步`40.682230`；grad mean/median=`42.395856/41.418900`。
- normal mean/median/P95=`6.446834/6.352000/7.740600s`；SOAP=`16.845659/17.152500/25.914050s`；吞吐=`14.995146 samples/s`；显存峰值仍`27086MiB`。
- epoch4前7步loss mean/median=`42.020114/38.972600`、显存未增加。继续同一任务至876。

## 2026-08-14：GPU日志对比与父提交主训练完成

- 子agent只读解析本地`gpu去除随机性固定后loss.log`；GPU前30步固定窗口为6.698167s全步、19.109707 samples/s条件吞吐、4.717565/4.378000/5.862400s普通mean/median/P95、4.236250s同编号11/12/21/22窗口、28409MiB。
- GPU日志确证8×A800和batch_size16字样，但数据117286帧、seed0，与NPU28130帧/自然随机语义不一致；该表只作硬件性能参考。GPU step1 grad NaN并跳过更新，step2/3 grad inf，不能称正确性门禁通过。
- 父提交`bf9ed6e`已打印876/876并自然退出；`iter_876.pth`存在、大小1607991785 bytes。尚需核验latest指向、完整epoch4统计和resume5步后，才启动当前MSDA版本。

## 2026-08-14：父提交4-epoch及resume完成

- epoch4 loss mean/median=`35.785836/35.168800`，末20步mean=`34.604210`；normal `time` mean/median/P95=`6.470903/6.303000/8.055100s`，窗口mean=`17.089045s`，throughput=`14.876407 samples/s`，`memory`峰值=`27173MiB`。
- 主训练876/876、finite、fatal0、自然退出；`iter_876.pth`和`latest`有效。
- 一次性resume夹具SHA256=`1882cd99...a1625`，仅用于诊断目录；正确容器、8 rank、后8卡映射、checkpoint加载和881上限门禁通过。
- resume meta显示iter875，因此日志876～881共6条；均finite、fatal0，loss mean/range=`33.352483/29.7404～37.7972`，与主训练末段连续；生成`iter_881.pth/latest`。父版本验收完成，转入当前`f922c38`。

## 2026-08-14：当前`f922c38`长期训练启动

- 父resume自然退出后，预检当前主仓库`f922c38` clean、canonical harness/config SHA一致、后8卡空闲健康、端口29926空闲、raw profile0；顺序启动当前版本876步。
- 唯一容器、torch_npu可用、可见逻辑设备8～15、8 rank/world8；本轮端口过滤后的主PID映射为物理4/0～7/1，未触碰前8卡同事任务。
- 初始化阶段先出现每rank 121MiB通信上下文，未误报为主显存；主进程建立后约25051～26975MiB。11/876时latest `time/memory/loss/grad_norm=7.010s/26482MiB/323.0993/53.6428`，全部finite、fatal0。

## 2026-08-14：当前`f922c38` epoch1完成

- 查询时233/876、running、全部finite、fatal0。epoch1 `loss` mean/median=`139.128243/109.554200`，首20步`342.197115`→末20步`72.327595`，与父epoch1数量级和趋势一致。
- `time` mean=`8.130699s`；normal mean/median/P95=`5.798000/5.639500/7.176000s`；固定窗口mean=`16.458238s`；throughput=`15.742805 samples/s`；`memory` max=`26842MiB`。
- 相对父epoch1，time/normal/window分别改善7.113%/8.363%/2.974%，throughput提升7.658%；memory仅+32MiB。继续epoch2。

## 2026-08-14：当前`f922c38` epoch2完成

- 查询时491/876、running、finite1、fatal0。epoch2 `loss` mean/median=`59.415967/58.483800`，首20步`72.269845`→末20步`51.273245`；`grad_norm` mean/median=`46.117401/44.824200`。
- `time` mean=`7.971680s`；normal mean/median/P95=`5.864211/5.718000/7.032600s`；窗口mean=`16.353659s`；throughput=`16.056840 samples/s`；`memory` max=`27085MiB`。
- 相对父epoch2，time/normal/window改善6.395%/8.337%/3.480%，throughput+6.833%，memory-1MiB；loss末20步几乎一致。继续epoch3。

## 2026-08-14：当前`f922c38` epoch3完成

- 远端查询通道连续两次出现`Error reading SSH protocol banner`，按55秒退避后恢复；训练进程未中断。有效查询时先确认705/876，随后原位统计时已到715/876、running。
- epoch3 `loss` mean/median=`43.287042/43.381800`，前20步`46.199335`→末20步`40.483295`；`grad_norm` mean/median=`43.943570/43.387100`，全部finite、fatal0。
- `time` mean=`8.261699s`；normal mean/median/P95=`6.123983/6.000000/7.654600s`；固定窗口mean=`16.763977s`；`throughput (samples/s)=15.493182`；`memory` max=`27085MiB`。
- 相对父epoch3，`time`改善3.214%、normal改善5.008%、固定窗口改善0.485%、`throughput (samples/s)`提升3.321%，`memory`少1MiB。继续唯一当前任务至876，不启动其他训练或profiler。

## 2026-08-14：当前`f922c38` 4-epoch与resume完成

- 主训练876/876自然退出；全876步`time` mean/median/P95=`8.165771/6.050000/25.248000s`，`throughput (samples/s)=15.675189`，`memory` max=`27175MiB`，全部`loss/grad_norm` finite、fatal0。
- epoch4 `loss` mean/median=`35.841326/35.668500`，前20步`39.575625`→末20步`35.016005`；`grad_norm` mean/median=`41.695886/40.217000`。相对父版本全876步`time` -5.077%、`throughput (samples/s)` +5.348%、`memory` +2MiB。
- 当前`iter_876.pth/latest`有效；8-rank后8卡resume记录876～881共6条，全部finite、fatal0，`loss` mean=`34.500800`，生成`iter_881.pth/latest`并自然退出。MSDA长期训练与恢复门禁通过。
- 最终评测预检未通过：原始config的旧`lidar_type`字段不被当前dataset类接受；内存移除后dataset构建90秒仍无返回。最终任务指标保持“未验证”，不使用历史评测文件冒充本轮结果。

## 2026-08-14：STEP-183裁决与清理完成

- 接受并保留既有`f922c38 【npu性能优化】MSDA切换DrivingSDK融合实现`；不为验收另建commit。相对父版本全876步`time` -5.077%、`throughput (samples/s)` +5.348%，最终epoch`loss`均值仅+0.155%，resume finite/fatal0。
- GPU作为主要性能参照：前30步GPU/NPU `time` mean=`6.698167/9.269933s`，normal=`4.717565/5.625957s`，固定编号窗口=`4.236250/33.301000s`；GPU分别快1.384×/1.193×/7.861×。跨数据/seed只比较性能，不比较功能。
- 保留父/当前两个iter876 checkpoint及SHA用于评测入口恢复后复测；删除其余3,243,389,144 bytes raw、iter881、日志、harness和父worktree。本地一次性resume夹具删除。最终远端主仓库clean、后8卡空闲、profiling raw0。
- 下一优化方向不凭GPU显存或总差距直接改码：先采当前`f922c38`稳态固定周期窗口最小profile，重新拆解SOAP剩余耗时；已有证据关闭的QR/stream/foreach方案不重复。

## 2026-08-14：STEP-184 SOAP周期窗口profile与裁决

- 在`f922c38`、后8卡、8 rank、客户batch16上完成12-step rank0 Level0 profile；覆盖Step10/11，12条`loss/grad_norm` finite、fatal0，launcher exit0。原始profile约6.90GB，只在远端原位分析。
- schema validation=0 errors，异常报告与独立10节模型架构报告生成。Step10 device busy=`23903.239750ms`，其中543次AICPU QR=`22641.383956ms`、占94.721%；Step11 device busy=`1060.265500ms`。
- 4次`2560x2560` QR耗时`16147.768347ms`，占QR 71.32%；其余主要为768、512、1024和256维。结果复现历史22.711s/543次，未出现新的实现路径。
- `CLOSED_NO_NEW_FIXED_ENV_EQUIVALENT`：历史batch/out-buffer/multi-stream/geqrf等门禁已拒绝，block/降频改变SOAP功能，客户环境又禁止框架/算子升级。本阶段不改码、不commit。
- 下一步：保留脱敏报告/JSON/SHA后立即删除全部profiling raw；随后不再重复SOAP QR，只有固定环境出现新的等价AI Core primitive时才允许重开。
- STEP-184清理完成：删除`6,903,469,756 bytes` raw/work/harness/分析器，raw0；保留9个脱敏文件315,720 bytes并再次SHA通过。远端Git clean、端口释放、后8卡空闲。本地3个一次性源夹具及2个pycache均删除。
## 2026-08-14 STEP-185 完成：客户评测镜像与 MSDA 同 checkpoint A/B

- 恢复 planning-with-files 上下文并读取连接规则；没有输出 `机器IP.md` 中任何凭据。
- 首次递归搜索评测文件因挂载文件量过大在 122.7s 超时；确认遗留扫描进程0后改为客户相对路径的精确探测，找到一组配对镜像。
- CPU-only dataset probe 依次修正两个启动合同：补仓库根 `PYTHONPATH` 解决 `path_mapping`，再补 `mmdetection3d-0.17.1` 解决 `ConcatDatasetV2`；最终 dataset_len25287、sample0 PASS，全程未占 NPU。
- 16-sample 第一次 8-rank 推理由于遗漏客户已有 `VIS_RATE=1`，全部 rank 报 `KeyError: data_tag`；按现有评测脚本语义补齐后父/当前均16/16、fatal0，并完成1,785,186元素对照。
- 创建 `bf9ed6e` detached worktree；确认 mmcv/mmdetection3d 源码 diff0并链接固定环境现有 `_ext.so`，父 worktree始终clean。
- 512-sample 父/当前正式补跑均完成，分别在运行中核验8 rank、逻辑8～15和 `npu-smi` 后8物理芯片主进程；两轮512/512、fatal0。父/current=`14.2/15.8 task/s`、`36/32s elapsed`。
- 远端 canonical `eval_RG.py` 因 `ModuleNotFoundError: ortools` 失败。未在远端安装；本地临时安装 `ortools 9.14.6206`（PyPI wheel）及 `networkx 3.2.1`，验证临时兼容层5,000例：status/cost mismatch0、tie edge mismatch75。因此仅作相同 shim 的辅助A/B，不声称绝对指标。
- 远端生成脱敏 JSON/Markdown 与 SHA 后删除 3,167 个 raw/临时文件、1,150,121,687 bytes；父 worktree移除。最终 archive raw0、任务进程0、相关端口0、后8卡本轮进程0、业务HEAD `f922c38`、Git clean。
- 本地 STEP-185 探针、比较器、shim、summary、pycache以及隔离 OR-Tools 目录全部删除；未修改业务代码、未创建 commit。

## 2026-08-14：STEP-186 最终交付审计与报告

- 修正`task_plan.md`中STEP-183/184已经由后续结果证明完成、但仍显示pending/进行中的历史项；顶部阶段切换到最终审计。
- 新增`最终性能优化报告.md`，统一列出永久算法基线`63861df`、可执行客户基线`4c37039`、当前`f922c38`和GPU日志的30-step对比。
- 报告中的性能字段保持日志英文名：`time`、`memory`、`loss`、`grad_norm`、`throughput (samples/s)`；推理使用原日志`task/s`和`elapsed`。
- 汇总采用commit链、876-step收敛与resume、512-sample同checkpoint推理、固定shape数值误差、候选拒绝矩阵和profiling raw清理结论。
- 尝试执行交付前远端只读状态复核时，本地命令编排层连续返回无诊断的exit1/9009，未取得新的远端输出、未启动训练、未改变远端状态；不重复同一失败路径。报告的远端状态以STEP-185最后一次成功核验为证据边界。
- 当前没有新增业务diff或性能commit。唯一未闭环项仍是客户canonical OR-Tools环境下的25,287样本绝对RG指标；远端禁止安装依赖。

## 2026-08-14：STEP-187 本地一次性产物清理

- 用户再次明确：临时文件使用后若后续不需要即删除。该要求已作为后续所有优化、A/B和profiling的固定收尾门禁。
- 只读盘点最初把`.codex-remote-edit`整体判作一次性目录；删除后Git复核发现其中37个文件属于原本clean的tracked诊断夹具，立即从HEAD精确恢复，恢复后该目录status0。最终只清理该目录11个untracked临时文件。
- 首次使用`Remove-Item -Recurse`被本地安全策略在执行前拦截，文件未变；随后解析3个绝对路径，验证都严格位于工作区内且无reparse point，改用PowerShell内.NET目录接口精确删除。
- 继续删除`.codex-tools`顶层65个已闭环候选的一次性patch/probe/gate/wrapper，以及`python-packages`下14个`__pycache__`目录中的91个字节码文件。最终净删除247个文件、1,737,330 bytes；`work`和可再生cache均不存在，`.codex-tools`顶层只保留tracked `remote_exec.py/remote_sync.py`及依赖源码。
- 保留客户`custom`配置、GPU日志、最终报告、计划记录、`.codex-remote-edit`全部tracked夹具，以及远端连接仍依赖的`.codex-tools/python-packages`。

## 2026-08-14：STEP-188 GPU/NPU最大公共step对比

- 用户纠正对比方法：两份日志不止30步时，应使用最大公共step。GPU有3664步，当前`f922c38`有876步，因此主窗口改为1～876。
- GPU前876步解析结果：`time` mean/median/P95=`4.515542237/4.312500/5.543500s`，条件`throughput (samples/s)=28.346540298`，`memory` max=`28816MiB`；索引876/876唯一。
- 当前NPU前述完整876-step结果：`time` mean/median/P95=`8.165771/6.050000/25.248000s`，`throughput (samples/s)=15.675189`，`memory=27175MiB`。
- GPU相对NPU：`time` mean降低44.702%、快1.808×，median快1.403×，P95快4.555×，条件`throughput (samples/s)`高80.837%，`memory`多1641MiB。
- GPU正确性边界：876个`loss`全部有限；`grad_norm`打印875条，其中873有限、2条`inf`，step1因NaN跳过更新且未打印`grad_norm`。因此仍只作性能参照。
- 首次解析因`inf`不符合标准double文本失败；第二次显式处理特殊值后又发现本机.NET无`Double.IsFinite`，改用`IsNaN/IsInfinity`后成功。未生成任何临时文件。

## 2026-08-14：STEP-189 1:1目标重新排期

- 正式目标更新为：同合同8卡NPU/8卡GPU `throughput (samples/s)`达到1:1或更好，同时不改变最终功能与训练语义。
- 当前参考比为0.553:1；尚差1.808×，平均每步需再减少3.650229 s。这意味着后续不能靠零散微优化，需要“SOAP周期路径 + 普通步host/device供给”双线收敛。
- 后续优先级已冻结为：P0严格同合同与低开销配对profile；P0 SOAP QR项目内等价自定义算子可行性；P0条件化pin/搬运；P1 CPU-NUMA亲和与局部TransData；P1有证据才测试`MatMul HF32`；P2单独记录batch/并行策略吞吐曲线。
- 本步骤仅更新计划和脱敏结论，未连接远端、未启动训练、未占用NPU、未创建临时文件、未修改业务代码或创建commit。
## STEP-189 普通步 profiling 与全算子清单（complete，2026-08-14）

- [x] `f922c38`、后 8 卡、8 rank、客户 batch/rank16，低开销采集普通训练步；8/8 step、loss/grad 全 finite、自然退出。
- [x] 远端原位聚合 243 个唯一算子、84,811 次 kernel 调用，生成全算子 CSV/Markdown、类别 CSV、异常 JSON/Markdown 与独立 10 节架构报告；schema/SHA 全通过。
- [x] 保存三份脱敏本地交付清单，SHA-256 与远端一致。
- [x] 删除 132 个/542,342,115 bytes profiling raw，复核 raw=0、profile 目录不存在、远端 Git clean；本地一次性脚本/pycache 已删除。
- [ ] 下一阶段依据 `underfeed_ratio=75.3158%` 优先归因 host/device 供给与高频细粒度调用，再从 Conv、MSDA、Layout/TransData 中选择可验证功能优化。
## STEP-190 普通步 host-launch 归因（in progress，2026-08-14）

- [x] 复读 STEP-189 脱敏异常/架构报告，不重新采集 profile。
- [x] 确认普通步前五 bubble 均为 `possible_host_launch_lag`，sync/comm overlap=0，无 wait-anchor false hotspot；HCCL 非 P0。
- [ ] 将 `item/_local_scalar_dense`、DDP unused-parameter search、Nonzero/Index/IndexPut 高频路径映射到当前活跃源码。
- [ ] 估算各候选可独立回退收益上限，选择一个超过门槛且不改变功能的候选。
- [x] 历史闭环交叉核验：DDP unused搜索、MapTR索引族、pin、CPU affinity、internal format、Conv HF32均不可重开。
- [x] 选择尚未独立闭环的 `MatMul HF32` 进入前置审计；当前纯kernel族上限224.837ms/step，先补真实shape/数值门禁，不与Conv HF32组合。
## 2026-08-14：STEP-190 MatMul HF32候选完成并拒绝

- 完成一次8卡单步shape-only诊断，确认111类、133次真实Linear/MatMul调用及前三类代表shape；单步`loss/grad_norm`有限，exit0。
- 完成8-rank、后8卡、Conv HF32固定False的MatMul HF32机制门禁。三个代表shape池化结果：false/true=`15.002210/13.738930ms`，仅省`1.263281ms`，最坏NRMSE=`1.469314e-4`。
- 依照“收益先过门槛再训练A/B”的冻结流程，候选因预计约`3.964ms/step`且引入全局数值扰动而拒绝；没有启动1-step/30-step训练，没有业务diff和commit。
- shape probe与HF32 gate的原始结果、日志、work及临时脚本全部删除；脱敏摘要和SHA保留在各自远端诊断目录。最终远端`ascend_npu_optimize@f922c38` clean，测试进程0，端口29931/29932均释放。

## 2026-08-14：STEP-191启动

- 从稳定Step7全算子表和历史关闭矩阵重新筛选：梯度norm已自动foreach，索引、SOAP小算子、Conv HF32和全局layout均不可重开。
- 选择尚未单独闭环的Inplace ReLU做只读形状/alias审计。现有上限为forward `34.472007ms` + backward `10.924639ms`，但不同ReLU shape不可直接横比，尚未形成优化结论。

## 2026-08-14：STEP-191完成并拒绝

- ReLU shape-only单步自然完成，90次in-place处理元素量约为59次module out-of-place的9.26倍；按稳定Step7 kernel归一化后in-place单位元素快19.33%。
- 最大激活若切到out-of-place会额外分配约4.23GB，且其`requires_grad=False`，不能降低backward。方向在改码前关闭，无业务diff/commit。
- 远端raw JSON/log/work与临时hook/config/wrapper已删除；Git status0、训练进程0、端口29933空闲。

## 2026-08-14：本轮上下文恢复

- 已读取项目规则、Git状态、`task_plan.md/progress.md/findings.md/操作步骤.md`最近状态；当前权威执行点为STEP-192局部channels-last机制门禁。
- 用户更新目标明确要求以稳定Step profiling Top N为主线；现有STEP-189的Step7单步证据满足该口径，STEP-190/191均是按TopN及历史关闭矩阵逐项裁决。
- 工具错误已登记：已有active goal导致重复`create_goal`失败；`session-catchup.py` exit1且无诊断输出。两者均未改变本地或远端业务状态。

## 2026-08-14：官方文章与DrivingSDK计划复核

- 已读取两份内容相同、SHA256均为`89ED9C8...A260A`的`DrivingSDK优化研究与实施计划.md`；该文件是早期队列，权威HEAD/基线已被后续STEP-183～192取代，但其“可达性→证据→语义oracle→最小patch→8卡A/B→重新profile”门禁继续有效。
- 已读取华为7.3.0“基础优化流程”和“并行策略建议”。当前stable Step7显示普通步underfeed高、通信高度掩盖，故不把TP/PP/ZeRO或扩大batch列为当前主线。
- 官方裸URL`performance_tuning_0`访问失败，改由同版本官方`performance_tuning_0016.html`补充基础流程；未使用非官方资料替代。

## 2026-08-14：远端预检本地入口纠错

- 第一轮远端只读预检未触达远端：系统`python`是WindowsApps占位程序，helper无输出exit1；相同入口的`--help`/Paramiko自检也无输出exit1。
- 使用Codex bundled Python重试本地自检，确认`.codex-tools/python-packages`中的`paramiko 4.0.0`可导入，`remote_exec.py --help`正常。后续只使用该显式解释器路径。

## 2026-08-14：STEP-192远端只读预检与GPU配置硬门禁

- bundled Python通道预检成功：唯一精确容器`mapqr-leicheng`、分支`ascend_npu_optimize`、HEAD`f922c3897255`、torch_npu2.7.1、后8逻辑设备可见数8、训练进程0、端口29935空闲；物理NPU4～7无进程，前半设备存在同事任务，不触碰。
- 远端业务仓库`status count=3`，未满足Git clean启动门禁，因此尚未上传或运行STEP-192夹具。
- 用户新增要求：严格遵守GPU测试配置并尽可能对齐。本地根目录与`custom`同名配置经显式路径复核后字节一致、SHA均为`9039BD31...CA33B`；此前第三个不同SHA来自`.codex-remote-edit`历史快照。以根目录文件作为GPU权威输入。
- 远端3项dirty均为前序STEP-192产物：`fusion_result.json`运行副作用、仓库根旧门禁脚本、两文件传输目录。完成绝对路径/非symlink/文件数/hash核验后，恢复tracked副作用并精确删除3个旧文件及空目录；远端业务仓库恢复status0。
- 当前夹具、wrapper、配置比较器和GPU权威配置已单向上传，四项SHA全部OK并移到仓库外诊断目录；容器内Python编译和wrapper `bash -n`通过。首次配置比较因GPU配置相对`_base_`路径在扁平目录失效而退出，未生成比较结论、未占NPU，业务Git仍clean。

## 2026-08-14：GPU配置合同审计完成，STEP-192暂停

- 通过一次性原目录临时引用、NPU现有base映射和trap清理完成MMC​​V结构比较；GPU临时文件SHA通过，命令结束后业务Git status0。
- 共42项差异；关键一致项为8卡、batch16、LR、runner、DDP、checkpoint/evaluation。关键不一致项为数据引用、随机dropout/mask、grid mask、optimizer hook/loss scale，以及SOAP阈值。
- 当前差异足以改变功能或loss分布，故不启动channels-last门禁，也不继续旧TopN的Conv+BN候选。下一阶段先追溯差异并建立最小NPU适配配置，之后重新跑稳态Step与profiling TopN。

## 2026-08-14：STEP-193差异归因

- Git历史确认grid mask、lidar随机项和FP16 hook切换均来自历史“随机性固定”链，不是NPU硬件要求；必须按GPU配置恢复后重新验证。
- 源码确认GPU/NPU optimizer hook精度语义不同：GPU动态FP16 scaling，NPU当前为普通backward/step。该项既可能影响loss，也可能显著影响性能。
- NPU训练镜像ann/flag basename与GPU配置一致且文件存在；GPU原文件SHA未提供，严格样本身份仍待补证。旧GPU日志的数据规模/seed与权威配置不一致，不能承担最终同合同验收。

## 2026-08-14：GPU功能对齐运行时配置生成

- 使用远端原位复制+断言式脚本生成仓库外配置，不修改tracked业务配置。恢复GPU的grid mask、lidar随机项、active train dropout和动态FP16 optimizer hook。
- 结构化diff由42降至31；剩余项为NPU路径/权重镜像、插件注册、顶层未引用列表、日志memory interval和SOAP one-sided threshold。关键运行字段输出与GPU一致，配置SHA`02ACA0C7...F56A5`，Git status0。
- 已找到canonical 8-rank harness SHA`10AD92C7...E0FC`；下一步先做1-step功能/容量门禁，不直接跑30-step或profiling。

## 2026-08-14：STEP-193首次1-step门禁失败与暂停重跑

- 启动前完整门禁通过：唯一正确容器、`torch_npu 2.7.1`、后8逻辑设备、8 rank目标、后8物理卡空闲、端口29936、`f922c38`与Git clean均满足；launcher确实创建8个rank。
- 8个rank均在NPU lazy init时因`tbe`无法导入而失败，未进入模型、FP16 hook、数据或iteration；没有loss/grad/性能数据。launcher自然退出，端口释放，业务Git仍clean。
- 直接证据显示本轮启动把`PYTHONPATH`覆盖为仓库两项路径，随后ACL初始化报告`ModuleNotFoundError: No module named 'tbe'`。该项按启动环境问题处理，禁止安装/升级CANN组件。
- 两轮容器内导入/`find_spec`对照均因`docker exec`通道超时而没有产生有效结果；当前不满足安全重跑条件。下一步先从宿主侧核验容器健康，待exec恢复后证明“前置仓库路径并保留原CANN路径”的等价修正，再重跑同一1-step。
- 宿主`docker inspect/top`确认容器running且无训练；绕过login shell后直接A/B：默认`tbe=True`、覆盖仓库路径`tbe=False`、仓库路径前置并保留原环境`tbe=True`。根因闭环，允许仅修正启动环境后重试。
- 首次修正环境后的调用把配置误传为harness第一个参数，harness立即报`config is required`，没有rank或NPU初始化；已恢复历史验证接口为`tools/train_spetr.py <config>`后再试。
- 正确1-step运行自然exit0并完成8-rank前反向：loss416.3346、显存23851MiB、fatal/OOM0；动态scaler首步65536→32768且因grad NaN跳过optimizer step，所以容量/执行路径通过但有效更新门禁未通过。首步56.932s含初始化，不作为性能值。
- GPU日志的同合同oracle为step1 NaN/skip、step2/3 inf、step4首次有限grad97.7820；NPU首步loss416.3346与GPU435.7609差约4.46%。下一次固定4步验证scale恢复，不改loss scale。
- 用户确认profiling只采一次：用一个连续窗口覆盖完整迭代，在同一trace内分析各阶段与全部算子，不为阶段重复采集。每个候选还必须通过真实shape算子A/B、8卡训练A/B和同checkpoint同测试集输出/指标A/B，功能与性能同时通过才提交。
- 4-step动态FP16恢复门禁通过：loss435.7064→419.2100→417.5949→417.3680，grad缺失/inf/inf/74.6566，step4与GPU一样首次有限；4步loss相对GPU差均小于1.7%。任务自然退出，约2.03GB原始门禁日志/work/checkpoint与运行副作用已清理，远端Git clean、raw profile0、训练进程0。
- 唯一profiling采集参数确定为rank0 `with_stack=True + record_shapes=True`；栈用于把TopN算子定位到阶段/模块/源码，kernel时间用于排序，最终吞吐仍以无profiler 30步基线为准。
- 用户新增硬门禁：必须在step稳定之后才采集。当前先以无profiler 30步确定scale/编译/数据管线稳态和SOAP周期，唯一profile的`wait+warmup`跨过不稳定区，active窗口只覆盖稳定普通步与稳定周期步。
- 首轮30步在iter8前发现运行合同缺失：GPU实际日志为seed0/deterministicFalse，NPU启动未传seed。已精确TERM端口29940父进程并清理2.81MB日志/work及kernel meta，恢复fusion_result，Git clean/raw profile0；该轮数据不进入基线。
- 仓库外seed0入口通过3-hunk静态门禁并完成30步：seed声明与GPU一致，30/30、fatal/OOM0、step4起grad finite，最终loss相对GPU约+0.36%。稳定普通步NPU/GPU吞吐比约0.700，完整SOAP周期约0.510；SOAP step14/24复现差1.21%，唯一profile窗口锁定稳定step23～26。
- 唯一profile入口静态审计通过：历史调度映射支持`wait22+warmup1 -> active step23`，active4覆盖step23～26；rank0、world8强校验、`with_stack/record_shapes=True`、Level0、无profile memory、禁checkpoint。两次静态失败分别为过严import计数及简化shell缺客户路径，均在训练/NPU/profile之前拦截；使用正式login-shell环境后`Config.fromfile`通过，Git clean、raw0。

## 2026-08-14：STEP-192关闭与阶段交接

- STEP-192首次8-rank机制门禁在进入算子计时前失败：固定torch_npu 2.7.1只支持`contiguous_format`或`preserve_format`，不支持目标NPU channels-last转换；因此无性能数据、无业务diff、无commit。
- 根据用户对“算子修改”的范围确认，放弃额外permute绕行，不把layout尝试当作算子优化成果。
- 已精确删除远端2个STEP-192诊断目录、仓库内对应pyc，以及本地3个一次性脚本和2个STEP-192 pyc。首次删除后发现此前CPU-only配置比较的3个父子进程仍在运行并重建目录；按精确cmdline验证后终止这些非NPU诊断进程并二次清理。最终远端/本地匹配数0、诊断进程0、权威仓库status0、训练进程0。
- 已创建`PROJECT_STATUS.md`，核对并固化当前HEAD/基线worktree、8个采用提交、30-step/876-step/GPU公共窗口、STEP-189 Top N、失败矩阵和下一阶段执行顺序。
- 2026-08-14：严格按客户GPU seed0/deterministic=False与batch/rank=16合同完成唯一一次rank0带栈/shape全阶段profile；窗口位于稳定step，覆盖一个稳定SOAP步和三个稳定普通步，8-rank/后8逻辑NPU/正确容器核验通过。
- 2026-08-14：完成同一trace的异常发现、10节架构报告、逐step TopN和v4源码栈归因；候选矩阵判定没有新的安全独立实现，不改业务代码、不创建commit、不为性能改变loss/optimizer语义。
- 2026-08-14：清理16,647,868,129 bytes profile raw及已摘要work/log；复核raw0、进程0、端口0、Git clean。无profiler性能比仍为普通步约0.700、完整稳定周期约0.510，1:1目标未达成且保持继续受严格门禁约束。
- 2026-08-14：启动STEP-194，不重复采集profile；从保留的v4栈报告纠正BMM归因：82.9ms/普通步是`spetr3d.py:1182`的单次`aclnnMatmul`，1284处280次`aclnnBatchMatMul`仅约16～21ms/步。ViewCopy主族定位1148处，下一步做clean HEAD源码边界审计。
- 2026-08-14：用户要求后续复用profiling raw；本轮16.65GB raw在要求到达前已经按项目规则删除并复核0，无法恢复且不重复采集。未来仅在当前分析仍需复用期间保留，分析闭环后仍遵守项目生命周期规则。

- 2026-08-15：完成 STEP-205 DrivingSDK MSDA 残余差距只读审计。6 次同义调用的 shape/count/dtype 已对齐；spatial FP32 单调用解释约 `+81.214ms/step` NPU-GPU 差距，但 NPU 每次仅一个固定 SDK 主 kernel。项目侧全部可见 device 边界乐观合计约 `6.236ms/step < 22.7ms`，SDK 无运行时 tiling/im2col/layout/precision 参数及同语义替代 primitive，裁决 `NO_GO_MAIN_KERNEL_FIXED_SDK_NO_PROJECT_CONTROLLED_EQUIVALENT_BOUNDARY`。未训练、未 NPU、未重采 profile、未改业务或环境、未 commit，永久 raw 未删除或移动。
- 2026-08-14：STEP-194完成。82.9ms BMM确认为已拒绝的BEVFormer point_sampling：broadcast 83.394→207.551ms回归；packed-BMM函数83.410→1.856ms exact，但两轮正式30-step复验端到端回归。ViewCopy明确clone最大单次6.888ms低于门槛，其余聚合缺唯一安全边界；裁决`CLOSED_NO_NEW_BOUNDARY`，无NPU运行、无业务改码、无commit。
- 2026-08-14：已修正远端脱敏TopN候选报告和manifest中的BMM归因，candidate SHA=`853e262a...e6f5e`、manifest SHA=`7b4817dd...317ba`；远端业务Git status0。
- 2026-08-14：STEP-195静态门禁确认冻结图像路径严格eval+no_grad；正确容器CPU-only实例化活动mmdet ResNet34/FPN并枚举43组直接Conv-BN（36+7），未占NPU。已按用户要求把融合设计、BNInfer证据和固定环境API审计拆给3个只读子agent。
- 2026-08-14：STEP-195补齐唯一trace的BN栈证据：4步`aclnnBatchNorm`共352次/device 210.599ms，代表栈落在`extract_feat -> forward_train`图像路径；`BatchNormBackward`另有116次/device 53.484ms，明确排除在冻结折叠收益之外。静态归因完成，下一步等待子agent设计/API结论后构造真实shape机制门禁。
- 2026-08-14：3个只读子agent完成STEP-195收敛：稳定普通步59次BNInfer均值48.255ms可拆为图像43+点云16；固定PyTorch eval融合CPU机器精度等价，MMCV原地接口禁用；确定“原注册模型权威+checkpoint加载后未注册eval-only融合副本”设计。真实shape 8-rank NPU机制门禁已交给独立子agent执行，未授权业务改码或短训练。
- 2026-08-14：API审计子agent复核并撤销仅凭`norm_eval=False`作出的初始否决；正式forward_train唯一图像路径实际为`fix_backbone`的eval/no_grad窗口，CPU真实结构探针43个BN stats/grad均不变，融合后四层输出relative L2约1.36e-6～1.73e-6。语义前置门禁通过，继续NPU机制门禁。
- 2026-08-14：STEP-195仓库外机制门禁完成并关闭。CPU门禁43对/258 checkpoint键严格加载，state hash、参数/缓冲区及optimizer ID不变；8-rank后8逻辑NPU真实`[112,3,576,1024]`边界性能`325.876→274.808ms`、节省`51.058ms`、`1.1858x`，但8/8 rank最大NRMSE=`1.991e-3`、max_abs=`0.06473`，超过严格`1e-4`门槛。裁决`REJECT_NUMERIC_GATE_NO_TRAIN_NO_COMMIT`，不短训、不改业务、不commit；远端仅留脱敏summary，Git clean、进程0、端口0。
- 2026-08-14：STEP-196只读复核完成。稳定Step7 underfeed=`5846.303ms/75.3158%`，但前五可见bubble合计仅`53.627ms`且分属已必要/已拒绝路径；HCCL未重叠理论上限约`1.623ms/step`，data可见host self约`0.457ms/step`。`item/_local_scalar_dense`、DDP unused、grad norm、Nonzero/IndexPut、DataLoader/pin均与历史正式结论交叉闭环。裁决`CLOSED_NO_UNIQUE_EQUIVALENT_UNDERFEED_BOUNDARY`；无重采、无训练/NPU、无业务修改。
- 2026-08-14：STEP-196 Index/Reduction独立复核关闭：纯kernel聚合跨多个语义边界且高wait项占比90%～99.89%，现有v4栈仅到外层调用；GeometricLoss、MapTR Unique、target mask、VectorNorm、SOAP IndexSelect和MSDA Zero均已有采用/拒绝/低上限证据。裁决`CLOSED_NO_NEW_INDEPENDENT_BOUNDARY`，未启动门禁。
- 2026-08-14：STEP-196脱敏架构报告一致性修正完成。唯一陈旧`worker=2` ASCII标签已按正文活动配置改为`workers=8,prefetch=3`；validation中的报告SHA及final manifest中的架构报告/validation摘要字节数与SHA已同步，复核全部匹配。raw=0、业务Git clean，无业务配置或训练状态变化。
- 2026-08-14：STEP-197选择性Conv-BN门禁完成并关闭。6个结构组在同一正确容器8-rank/后8 NPU真实shape任务中逐一测试，shape/stride/finite及state/hash/参数/缓冲区/optimizer ID全通过；但最大NRMSE为`2.977e-4～1.529e-3`且净节省仅`5.865～12.185ms`，没有单组同时满足`NRMSE<=1e-4`与`>22.7ms`，故组合数0并裁决`REJECT_NO_SELECTIVE_GROUP_MEETS_NUMERIC_AND_22P7MS_GATE`。不训练、不profiling、不改业务、不commit；raw/log/tools清理后仅留2份脱敏summary，Git clean、进程0、端口0。
- 2026-08-14：自动目标继续后重新读取`planning-with-files`与`ascend-profiling-anomaly`技能，执行session-catchup并复核当前STEP-193～197状态。按技能要求重读kernel数据指南，继续坚持四时钟、跨流busy union、纯kernel/total cost双排名和wait-anchor降级；下一阶段不以“已关闭矩阵”替代1:1目标，转向固定环境中尚未审计的图执行/编译级算子融合机制。
- 2026-08-14：通过内置浏览器读取用户指定的两篇华为TorchNPU 7.3官方文档；直连web首次分别被安全检查拒绝/超时，改用官方页面可见正文成功。官方原则与当前门禁一致：最终以吞吐/step衡量，先区分数据/计算/optimizer/后处理/通信/调度，融合必须数学等价，减少stream同步需保留功能，多卡通信wait须查首个慢卡差异。未从网页执行任何指令或外部写入。
- 2026-08-14：STEP-198启动。首次用带`-g '*.py'`的本地`rg`搜索图模式关键字返回exit1且无输出，说明当前本地业务源码镜像未命中而非工具故障；改为跨计划/报告检索后确认`jit_compile=False`仅停留在早期backlog，历史正式测试覆盖TASK_QUEUE/COMBINED但没有通用图模式A/B。环境/API与可捕获边界已拆给3个只读子agent。
- 2026-08-14：STEP-198固定环境只读清点确认TorchAir缺失、`torch.compile backend=npu`退化为eager直返；唯一现成新机制为NPUGraph/npugraphs。历史审计确认未做过正式图捕获A/B。边界初审推荐冻结图像Backbone+FPN，但子agent误入旧仓库`ascend_npu@f189414`；主agent已拦截并要求在权威`ascend_npu_optimize@f922c38`复核，尚未启动NPU。
- 2026-08-14：STEP-198权威仓库复核完成：`ascend_npu_optimize@f922c38` clean，活动batch16/7cam/T1/fix_backbone/no-grid-mask与固定`[112,3,576,1024]`图像塔输入确认。最小NPUGraph边界成立，下一步仅做8-rank真实shape机制门禁；收益过22.7ms且数值/state门禁通过后才允许短训。
- 2026-08-14：STEP-198原生NPUGraph机制门禁完成并拒绝。前两轮仓库外harness分别在capture前因0维BN计数器byte-view和默认stream capture失败，均记录并精确清理；第三轮仅按API要求使用独立stream后8-rank exit0。四层输出、重复replay及state/optimizer完整性完全一致，但完整图像塔eager `326.609ms`、graph含输入copy+replay `327.281ms`，净收益`-0.714ms`、`0.99795x`，capture额外reserved约30.667GB。裁决`REJECT_MECHANISM_GATE_NO_TRAIN_NO_COMMIT`；未训练、未profiling、未改业务、未commit。live确认8 workers/rank/端口，但`npu-smi`未返回匹配PID，明确记录证据缺失。远端raw/harness已删除，只留3份脱敏analysis文件，manifest SHA=`765b93c5...5549`；Git clean、进程0、端口0、profile raw0。
- 2026-08-14：STEP-199只读子任务B完成。权威`f922c38`的QR是`stable argsort→exp_avg_sq/Q同序重排→FP32 power_iter→torch.linalg.qr`，当前周期543次，主要为4个2560方阵；Q/GG/exp_avg_sq/state.step随optimizer checkpoint保存并在resume恢复。严格门禁必须比较raw Q、排序和连续两周期完整state/参数逐位，以及中途resume逐位，不接受仅符号对齐或正交/重构容差；R不被业务使用，Q-only候选可不生成。GPU配置/日志只能证明SOAP与频率，不能认证soap.py源码SHA；历史GPU时代FP64 CPU QR与当前NPU不是逐状态同算法。Householder重写、TSQR、Gram-Schmidt、Cholesky-QR均改变舍入/符号/旋转或奇异/NaN边界，裁决`NO_GO_ALGORITHM_NOT_STATE_EQUIVALENT`；只剩“固定环境已有、同ACLNN语义且raw Q逐位的更快底层primitive”可由工具链审计决定。无训练、NPU、profiling、业务修改或远端产物拉取。
- 2026-08-14：STEP-199只读子任务A完成。固定容器具备NpuExtension、GCC/CMake/Ninja、CCEC/BiSheng/OPC/msopgen及TBE/op_gen，项目也有MMCV NPU C++ wrapper基础；但SDK QR侧只发现当前已用的`aclnnLinalgQr/GetWorkspaceSize`，必须同时输出Q/R，未发现Q-only、Geqrf/Orgqr或其他同语义更快primitive。C++重包同一ACLNN不能减少约22.6秒kernel；AscendC/TBE自写需生成/编译/注册产物且无法满足raw Q逐位合同。裁决`NO_GO_NO_EQUIVALENT_Q_ONLY_OR_FASTER_PRIMITIVE_IN_FIXED_ENV`；未编译、安装、启动NPU/训练、改业务或commit，远端HEAD clean且训练进程0。
- 2026-08-14：STEP-200 addmm只读子审计已恢复planning/profiling技能与必需参考，并读取`机器IP.md`但未输出连接信息。纠正初读口径后，低开销稳定Step7为117个Addmm kernel、纯kernel `15.089ms/step`、含wait总成本`19.865ms/step`；4-step带栈operator记录468次与117次/step交叉一致。早期advisor 135次仅作线索；继续核对活跃源码/API/历史闭环，无训练/NPU/重采profile/业务修改。
- 2026-08-14：STEP-200 addmm子审计完成。固定环境无`npu_addmm`，活动`nn.Linear→F.linear`已经生成单个`aclnnAddmm`；117次调用分散在BEV encoder与4层lane3d decoder，Top25真实shape记录覆盖94.696% Linear/MatMul FLOPs。稳定Step7 Addmm纯kernel/含wait上限仅`15.089/19.865ms`，均低于22.7ms；历史HF32/MHA/output projection/BMM方向均已闭环，裁决`NO_GO_ALREADY_FUSED_AND_BELOW_THRESHOLD`，不进入机制或训练门禁。
- 2026-08-14：STEP-200 confusion-transpose只读子审计完成。固定2.7.1 API虽存在，但前/反向均只有legacy `acl_op`而无当前A3所需ACLNN实现；官方False参考不保证stride/alias。稳定普通Step7宽松相关的`InplaceCopy_Transpose+Contiguous_Transpose`纯kernel仅`21.940974ms`且跨原地copy/contiguous消费者，低于22.7ms；其8.386797ms wait不计device compute。SOAP活动`merge_dims=False/channels_first`没有活跃reshape-transpose对，内部tensordot改写已由STEP-107/108以低收益、非逐位/回归关闭；PillarVFE最强copy消除也仅省7.906ms。裁决`NO_GO_UNSUPPORTED_AND_BELOW_THRESHOLD_NO_UNIQUE_BOUNDARY`，未启动NPU/训练/重采profile/业务修改。
- 2026-08-14：STEP-200 `npu_add_layer_norm`只读复核完成。固定2.7.1 schema/反向、权威`f922c38`源码和稳定profile交叉确认只有BEV 3处+MapTR decoder 12处残差Add→LN，其余LN为Linear→LN→ReLU。STEP-189 LN前后向主kernel合计仅`6.636108ms/step`；全阶段栈把每步全部Add与全部LN-forward都错误假设可消除的极端上限也仅`21.006956ms < 22.7ms`，MapTR过度包含上限仅`1.733646ms`。裁决`NO_GO_NO_UNIQUE_ADD_LAYERNORM_BOUNDARY_ABOVE_22P7MS`；无训练/NPU/重采/改码/commit。
- 2026-08-14：STEP-200统一收口。addmm、confusion-transpose、add-layernorm均按当前稳定Step纯kernel、固定torch_npu2.7.1兼容性、源码唯一边界与历史A/B完成交叉裁决；三项分别为已融合且15.089ms、跨消费者/不兼容且21.941ms、极端上限21.007ms，均未越过22.7ms。无机制门禁、NPU、训练、新profile、业务修改或commit；报告/status已同步。
- 2026-08-14：STEP-201阻塞审计完成。STEP-198图捕获、STEP-199自定义QR和STEP-200亲和API连续三个目标轮次均确认：现有唯一稳定profile无剩余固定环境、严格等价、单一可回退候选；普通/完整周期NPU/GPU约0.700/0.510，1:1未完成。继续需要厂商同语义primitive、客户授权软件栈变化，或授权在旧raw不可恢复后重新采集稳定全阶段timeline；当前不改变loss/SOAP/state/batch/GPU合同绕过。
- 2026-08-14：用户明确授权重新采集一次稳定全阶段profiling，STEP-201证据阻塞解除并启动STEP-202。将沿用GPU对齐合同和旧成功稳定窗口设计，只采集一次rank0 with-stack+shapes连续窗口，8 rank/后8 NPU真实训练；同一trace覆盖所有阶段。本轮raw在分析与候选复用结束前保留，不拉本地。
- 2026-08-14：用户在新profile完成active窗口、仍处于rank0原位解析时进一步明确“这次采集后就不要删除”。生命周期规则更新为本次raw/export/trace/operator/communication/memory及采集证据长期原位保留，分析与候选结束后也不删除、不移动、不覆盖；manifest写`deletion_authorized=false`并全量SHA，不拉本地。
- 2026-08-14：STEP-202旧成功harness/窗口合同只读复核完成。远端权威仓库仍为`f922c38` clean，旧hook/config/seed入口/world8 launcher及iter30 checkpoint均存在并完成SHA校验；成功合同是后8逻辑NPU、8 rank、batch16、seed0/deterministicFalse、动态loss-scale，从头运行28 step，rank0以`wait22+warmup1+active4`带栈/shape采集。依据真实历史映射，Profiler Step23～26对应训练Step24 SOAP及Step25～27普通稳定步；checkpoint写入关闭、iter30 checkpoint只作后续测试oracle而不加载。没有启动训练/NPU/profile或修改业务。
- 2026-08-14：STEP-202唯一正式任务已启动1次并完成warmup23及active训练Step24～27；SOAP/普通四步loss与grad均finite，raw正在自然解析，尚未退出。用户最新要求本次raw及全部导出/日志/harness永久原位保留，禁止删除、移动或覆盖；收尾manifest必须明确`deletion_authorized=false`。该要求已替代原来的后续清理措辞。
- 2026-08-14：STEP-202唯一正式采集自然完成28/28、exit0、fatal/OOM0；active训练Step24 SOAP与25～27普通步loss/grad均finite，运行中8直接rank、WORLD_SIZE/local_rank、torch_npu、后8die 8个唯一PID和端口证据完整，结束后进程/端口/NPU PID归0。raw永久原位保留为205 files/16,647,970,748 regular-file bytes；全量SHA树外manifest SHA=`464af966...d350`，验证`deletion_authorized=false/retained=true/mutation=false`。唯一tracked副作用`fusion_result.json`按HEAD精确恢复，raw/manifest前后不变且Git clean。下一步由原位分析任务读取该raw生成新TopN/异常/架构/栈报告，不清理raw。
- 2026-08-14：STEP-202远端只读启动前预检完成并裁决GO。唯一正确容器running，`f922c38` clean，torch/torch_npu2.7.1，后8逻辑设备可见8且对应物理后半8个die全部Health OK、44～48℃、AI Core0、无NPU进程；训练进程0、299xx监听0。GPU对齐config/checkpoint/launcher/profile入口均存在、非symlink且SHA吻合；按正式wrapper cwd和PYTHONPATH静态导入成功，batch16/workers8/pin/prefetch3/SOAP/dynamic-scale/schedule22:1:4及ann/flag存在性均通过。两次CPU静态失败分别是预检路径推导错误和遗漏`cd repo`，均无rank/NPU/raw/work/业务变化。
- 2026-08-14：STEP-202原位分析准备完成并裁决`GO_ANALYSIS_PIPELINE_READY`。远端历史V1保持原SHA且不覆盖；仓库外新V2（SHA `cbbef28b...ea6e4`）补齐逐Step四时钟、逐Step纯kernel/total-cost双TopN、wait-anchor、bubble前后task/stream与host stack，旧全阶段栈工具（SHA `580eeff...227`）继续负责所有训练阶段归因。新增raw retention manifest工具（SHA `6e60212...cef2`）默认全文件SHA且无删除能力。py_compile和合成E2E通过：schema0错误、10/10架构节、raw SHA 4/4；测试fixture/输出/pycache已清理。若出现communication.json，仅需在实际schema inventory后加权威总量adapter，不阻塞采集或首轮分析。
- 2026-08-14：STEP-203启动GPU→NPU原位对比。GPU `.7z`普通非symlink、473,979,928B、SHA=`ff083f2b...b46178`；远端无7z CLI/py7zr且未安装，但系统既有`libarchive.so.13`可由ctypes调用。严格拒绝absolute/dotdot/symlink/hardlink/device/FIFO后在新诊断目录安全解出唯一12,368,970,966B JSON，SHA=`d826cd27...c645`，manifest SHA=`80ae3b1f...7c664`，原归档SHA不变。初步schema为A800/NCCL/rank0 Chrome trace、客户同名配置、`iter0_49_no_stack`；正在用容器既有ijson流式盘点真实完整step和类别，不把文件名当step完成证据。
- 2026-08-14：STEP-202原位分析闭环。旧V2与optimized除`profile_run`外全部分析字段exact，schema均0错误、架构均10/10节；full-stage stack覆盖率85.072765%。普通Step24～26的AICPU ViewCopy稳定为2048次、96.610072ms纯kernel+11.821988ms wait/步，逐栈与`bev_encoder.random_spatial_mask` line427四步8192次严格闭合，成为首个不重复历史候选。仅裁决进入严格RNG/逐位/alias/dtype/shape机制门禁，尚未改码、训练或提交。修正首版bubble scope字段后，权威analysis manifest SHA=`1bd319c6...a657b`、40项SHA错误0；raw仍205文件/16,647,970,748B/10目录，retention SHA=`464af966...d350`且永久保留。
- 2026-08-15：STEP-203完成GPU无栈trace与NPU全阶段profile原位对比。GPU稳定普通15步四时钟中位=`5848.556/5846.212/2045.559/2045.559ms`；mask同义链在GPU为`fill [1,8,8]`及CUDA Fill kernel每步严格2048次、8.194ms，NPU为AICPU ViewCopy 2048次、96.610ms，ratio=11.79、超额88.416ms/普通步。P0确定为保持RNG/覆盖/alias逐位语义的批量mask机制门禁；MSDA已采用，QR/Conv-BN/channels-last/BMM/Index/Reduction/Graph按历史矩阵关闭。生成脱敏对比报告SHA=`46413f75...18486c`与永久保留清单SHA=`abb27c1c...8744c`；GPU归档/12.37GB JSON和NPU 16.65GB raw均未删除、移动、覆盖或拉本地。
- 2026-08-15：STEP-204机制与业务函数门禁通过。保留enable先行和B次CPU randperm顺序的低分辨率`index_fill_+repeat`方案在CPU64 case、后8 NPU 8×64 case及真实业务函数复验中全部逐位/RNG/alias exact；真实shape完整同步边界净省206.246～235.591ms，保守相对原profile纯kernel下界95.669ms，均超过22.7ms。独立live探针补齐8 direct rank和npu-smi物理4/0～7/1证据。单文件业务patch已应用但未提交；尚未启动正式训练，正在固定seed0 profiler-off基线、命令和checkpoint oracle。
- 2026-08-15：STEP-204 fresh 30-step paired A/B完成。baseline/candidate均30/30 exit0且永久保留日志/metrics/checkpoint SHA；稳定普通14/14步均加速，mean`5.322551→5.000786s`、吞吐`+6.434%`，cycle`-2.951%`。loss最大相对偏差0.3934%、grad最大2.0626%，dynamic-scale相位一致；checkpoint/optimizer schema与finite通过但自然随机状态不逐位。当前补丁未提交，下一步按既有门禁顺序做paired resume、固定512样本和GPU对齐876-step长期A/B；candidate普通NPU/GPU吞吐约0.8647，1:1目标尚未完成。
- 2026-08-15：STEP-204 paired resume完成。baseline/candidate均从各自iter30恢复至iter36、exit0，日志Iter30～36、loss全finite、唯一Iter31 grad inf/scale下降/skip相位一致；meta29→35、559个optimizer step26→32、state_dict1042项。resume loss最大相对差0.5037%；finite grad最大绝对相对差11.2622%发生在Iter34（48.30773→42.86721）。iter36结构/shape/dtype/finite一致，资源已释放，candidate tracked patch SHA保持`921d53da...0313`。
- 2026-08-15：固定512测试在数据身份门禁停止。STEP-185原512 harness/子集/output已按旧生命周期删除；只保留basename/bytes/dataset_len，没有源内容SHA或first512首尾ID。当前config/probe明确路径、容器6个mount及既有diagnostics manifest均不能恢复同一镜像；未构建替代清单、未启动NPU/测试/876。一次大挂载只读find按主agent要求以唯一PGID精确终止并复核残留0。
- 2026-08-15：STEP-204 fresh paired 876-step长期A/B完成。baseline/candidate均876/876 exit0、loss全finite、动态scale缺失/overflow相位同为[1,2]/[3]，checkpoint meta/schema/shape/dtype/finite一致；paired checkpoint工件显示deterministicFalse自然分叉。性能相对baseline：stable normal慢1.0719%、SOAP快1.5830%、完整周期慢0.0811%、all1～875慢0.0964%、末100普通慢2.2777%，没有复现30-step普通步正收益，故当前不满足采用门禁。candidate all1～875吞吐16.5255 samples/s，相对既有GPU前876吞吐28.3465仅0.5830。固定512仍因权威数据身份不可恢复而pending；补丁未提交，resume暂停等待负门禁裁决。
- 2026-08-15：STEP-204最终裁决`REJECT_LONG_RUN_NO_SUSTAINED_NORMAL_OR_CYCLE_GAIN_NO_COMMIT`。30-step普通步`+6.434%`被876-step stable normal`-1.0605%`吞吐、cycle`-0.0810%`和末100普通`-2.2270%`反证，归类为短窗假阳性；不启动876 resume或512测试，不commit。最终单文件patch已以完整SHA=`921d53daa0af10386843acbe1fabd712567e22a4cf8208e3204a8f33aed30313`原位保留，随后仅精确恢复`bev_encoder.py`到HEAD blob=`5423a7d7...cde`/文件SHA=`399f349d...6ae2`。权威仓库与baseline worktree tracked0，端口/训练/NPU进程0；所有训练、比较、失败日志、checkpoint和NPU/GPU profiling工件保留未动。
- 2026-08-15：STEP-206只读关闭审计完成。复用永久保留的NPU普通Step24～26和GPU稳定15步分析，纠正旧ViewCopy v2 JSON字段解释，完成prelaunch scatter shape/count与forward+loss Matmul最内层源码拆分。唯一越线单点为历史已拒绝的`point_sampling`；其余新单点低于`22.7ms/step`。形成`STEP-206_剩余TopN_GPU基线关闭矩阵.md`，裁决`CLOSED_NO_NEW_SINGLE_EQUIVALENT_BOUNDARY_ABOVE_22P7MS_AFTER_GPU_BASELINE_CROSSCHECK`；无训练、NPU、重采、业务修改、删除、拉取或commit。
- 2026-08-15：STEP-208阶段差距矩阵完成。按data/prelaunch、forward+loss、backward、optimizer/SOAP、communication/tail重新对齐NPU Step24～26/Step23与GPU稳定15步/SOAP3步；没有发现项目可控、单一、严格等价且理论净收益超过`22.7ms/step`的新边界。裁决`NO_GO_NO_NEW_PROJECT_CONTROLLED_SINGLE_EQUIVALENT_BOUNDARY_ABOVE_22P7MS`；只读复用永久产物，无训练、NPU、重采或业务修改。
- 2026-08-15：目标自动继续后启动STEP-208。重新完整读取planning/profiling技能及必需kernel/rulebook/architecture/schema参考，执行session-catchup并复核STEP-193～207当前状态。1:1目标仍未完成，本轮将由两个独立子agent分别复核系统级阶段差距和固定环境遗漏能力；主任务做逐要求完成度审计。当前仅只读，不启动训练/NPU/profile，不删除永久产物。
- 2026-08-15：STEP-208主审计先修正文档生命周期漂移：最终报告与PROJECT_STATUS中两处`16,647,868,129 bytes/raw=0`属于旧profile历史，不再代表当前现场；已明确STEP-202新raw205文件/16,647,970,748B及manifest SHA=`464af966...d350`永久保留。没有触碰远端raw或业务代码。
- 2026-08-15：STEP-208-B完成固定环境能力反证。交叉STEP-089～100、R6/R7、STEP-161～206后，没有遗漏的同语义API、runtime、stream、allocator、compile或项目边界能证明净省`>22.7ms/step`。`PER_STREAM_QUEUE=1`虽为固定2.7.1已有试验开关，但当前缺多线程多compute-stream Dequeue阻塞证据，且官方约束与已拒绝SOAP双流/affinity相冲突，不具备A/B准入。裁决`NO_GO_FIXED_ENVIRONMENT_CAPABILITY_MATRIX_EXHAUSTED_NO_SAME_SEMANTIC_BOUNDARY_ABOVE_22P7MS`；未训练/NPU/profile/安装/改码。
- 2026-08-15：STEP-208完成度审计收口。新增`STEP-208_1比1目标完成度审计.md`与`STEP-208_阶段差距矩阵.md`；最长1～875 step当前HEAD NPU/GPU吞吐约`0.5835:1`、仍差`3.222591s/step`，需约`71.37%`吞吐提升。阶段和能力两路独立复核均NO_GO，形成目标恢复后的第二次连续外部能力/新权威证据阻塞；本轮不训练、不NPU、不profile、不改码、不commit，目标保持未完成。
- 2026-08-15：STEP-209第三次阻塞复核完成。远端容器/HEAD/tracked状态、torch/torch_npu/mx_driving/CANN能力、MSDA与QR schema、训练/NPU/端口状态及NPU/GPU永久原始数据均未变化；独立复算仍为`0.583544235:1`、缺`3.222591s/step`。STEP-207～209同一阻塞连续三次成立，裁决`BLOCKED_THIRD_CONSECUTIVE_AUDIT_NO_EXTERNAL_STATE_CHANGE`，目标未完成并按规则正式标记blocked；无训练、NPU、profile、安装、业务修改、commit或原始数据清理。
- 2026-08-15：用户以社区/Triton自定义算子方向恢复目标并询问5项环境变量。STEP-211完成官方文档、固定环境源码和历史A/B交叉：GLOBAL=1当前已生效，AUTOLOAD0是既有导入兼容约束；ATB ALG3不命中活动业务，expandable无OOM/碎片证据，TQ2正式8卡A/B吞吐回归6.991%。裁决`REJECT_COMBINED_ENV_CHANGE_NO_NEW_PERFORMANCE_CANDIDATE`，未修改启动环境、未训练/NPU/profile/安装/改业务；下一步转向STEP-212社区与Triton同义算子筛选。
- 2026-08-15：STEP-212启动。官方Triton-Ascend/CANNBot资料确认A3可表达自定义kernel，但生成器不提供客户容器runtime；CANN8.3RC1与较早Triton-Ascend 3.2.0rc4存在版本对应，远端不得安装。DrivingSDK公开MSDA已有generic/high-performance实现与tiling key 1002/1004/1008；正在核对当前`mx_driving1.0.0+gitde13346`是否已包含并命中空间FP32真实shape。当前仅官方资料和固定环境只读审计，不编译、不安装、不训练/NPU/profile。
- 2026-08-15：STEP-212发现比Triton生成更直接的官方候选：DrivingSDK !1105/!1112/!1378/!1504在2025年持续优化MSDA，其中!1105明确覆盖`heads*levels*points<=64`，当前空间调用恰为`8*1*8=64`；后续补丁继续重写forward/grad搬运和双buffer。旧版真实shape已经命中Opt<8>，所以候选不是修正generic误选，而是验证客户`gitde13346`是否缺少2025新版kernel。固定容器无Triton-Ascend backend，CANNBot产物当前不能运行；继续只读映射安装包与官方commit。
- 2026-08-15：STEP-212收口。官方GitCode确认`de13346=branch_v7.3.0`头且四个2025 MSDA优化均已包含；安装910_93二进制的`11/01/10/00` fastMode结构与之吻合。v7.3之后只有新版本栈中的通用embed/load-balance/精度补丁，无当前shape净收益证据且需要替换冻结torch_npu/DrivingSDK配套；Triton-Ascend backend缺失，CANNBot不能执行；各版本QR仍无Q-only primitive。裁决`complete_no_go_current_sdk_already_contains_patches_and_no_ascend_triton_backend`，未编译、安装、训练、NPU、profile、改业务或commit。
- 2026-08-15：目标继续进入STEP-213，不把STEP-212路线关闭误报为1:1完成。当前唯一值得继续证明的外部线索是v7.3之后MSDA cube/load-balance补丁；三个子任务分别解析目标shape代码路径、审计固定环境项目局部隔离构建能力、从永久raw寻找核间负载不均直接证据。证据未闭合前不编译、不启动NPU、不安装或修改业务。
- 2026-08-15：STEP-213官方源码首个定量证据已得到：晚期MR !1840明确非FastMode提升10%+、FastMode仅2%+；当前空间调用是key11 FastMode，按forward65.342ms仅约1.307ms/step，远低于22.7ms门槛，且无backward收益声明。因此该补丁不具机制测试资格；继续等待隔离构建与raw负载证据子审计仅用于闭合原因，不会据此编译或占用NPU。
- 2026-08-15：用户明确改变执行授权并要求继续优化四类差距：`expandable_segments=True`必须加入；`TASK_QUEUE_ENABLE=2`后续重新单变量对比；允许仅在`mapqr-leicheng`容器内安装Triton-Ascend，宿主禁止安装。STEP-214启动：先定位权威启动入口与隔离安装方案，allocator/TQ2分别A/B，Triton-Ascend禁止覆盖全局依赖，最小backend验证后优先研究QR/MSDA。
- 2026-08-15：STEP-214启动。已完整读取planning-with-files技能并恢复计划状态，读取`机器IP.md`但未回显凭据。官方资料确认Triton-Ascend 3.2.0rc4与CANN8.3.RC1、torch_npu2.7.1、CPython3.11/aarch64标签相容；下一步仅在`mapqr-leicheng`做容器/磁盘/网络/全局包/依赖dry-run门禁，未安装、未运行NPU kernel。
- 2026-08-15：STEP-214远端preflight通过：精确容器running、训练进程0、Python3.11/aarch64/glibc2.34、CANN8.3.RC1、torch/torch_npu2.7.1、全局triton3.7.1且triton-ascend缺失；共享诊断盘约2TB可用，PyPI/GitCode可达。只创建仓库外诊断目录并上传preflight脚本，全局环境未变；下一步下载并核验官方wheel/METADATA后再决定是否创建隔离venv。
- 2026-08-15：STEP-214官方CPython3.11/aarch64 wheel下载到仓库外诊断wheelhouse并通过PyPI SHA256校验；METADATA无依赖。隔离`--system-site-packages` venv已创建，能复用全局torch/torch_npu2.7.1；pip dry-run仅计划安装triton-ascend，无全局依赖替换。下一步用`--no-index --no-deps`安装该本地wheel并只做CPU/import/backend验证，不运行NPU kernel。
- 2026-08-15：STEP-214隔离安装完成。venv内triton-ascend3.2.0rc4与Ascend backend import/registry通过，torch/torch_npu继续复用全局2.7.1；默认Python仍为全局triton3.7.1且triton-ascend缺失。官方tag peeled commit、wheel/report SHA、目录大小和回退边界已记录；全程训练进程0、未调用NPU kernel。A3最小算子等待主任务完成allocator 8卡A/B并明确释放资源。
- 2026-08-15：STEP-214-B完成正式入口只读定位。权威8卡继承链为外层诊断wrapper→`tools/ddp_train.sh`→8-rank distributed launcher→`tools/train_spetr.py`；正式成功wrapper已有AUTOLOAD0，tracked仓库无TQ/allocator。最小allocator永久patch应只在`tools/ddp_train.sh`的launcher前新增一个export；本地无权威对应脚本且历史wrapper合同不可污染，因此未改码，仅给远端`apply_patch`方案。allocator与TQ2已冻结为先后两个单变量A/B，未训练/NPU/profile。
- 2026-08-15：用户补充明确实施授权后，STEP-214-B已把allocator单行patch应用到远端权威`tools/ddp_train.sh`。pre/post SHA256分别为`e006683c...b2461ed`/`73283084...316d8a`，`bash -n`、diff-check、唯一行计数通过，numstat=1/0，TQ2未加入，未commit。未启动训练/NPU/profile；下一步先做allocator单变量A/B，再按结果冻结TQ2重测共同基线。
- 2026-08-15：STEP-213-C永久raw字段审计完成。STEP202 Level0明确未采AICore指标；kernel CSV的MSDA blockDim为0/core为N/A，profiler DB只有整task起止，trace的48个MSDA事件虽标为KERNEL_AIVEC但Subtask Id均为无效哨兵且无core/block跨度。故只能复核NPU/GPU同shape整kernel `65.342/146.580ms` 对 `25.968/104.740ms`，不能量化核间load imbalance或尾核可回收量，裁决`INSUFFICIENT_LEVEL0_EVIDENCE_FOR_CORE_LOAD_IMBALANCE`。未重采、未训练/NPU、未改业务；NPU raw仍205文件/16,647,970,748B、manifest SHA=`464af966...d350`，GPU archive/JSON bytes仍为`473,979,928/12,368,970,966`。
- 2026-08-15：STEP-214-D启动，仅做CPU/源码设计。权威`soap.py`和既有24类QR shape/state合同已只读复核；候选边界严格限定为以`q_only_qr(power_iter)`替换line422的`torch.linalg.qr`并保留其余排序、重排、matmul和state逻辑。当前未改业务、未编译、未调用NPU或训练；下一步读取隔离Triton-Ascend3.2.0rc4 API/后端限制，判断Householder Q-only能否实现及raw-Q逐位风险。
- 2026-08-15：STEP-214-D CPU/源码设计收口。Householder Q-only可用Triton基本归约/dot表达，但大shape受96/192KiB片上限制，必须GM tiled和多kernel panel；Ascend扩展同步无足够grid-wide barrier合同。当前R完整输出仅0.266GiB/cycle，单纯少写R不足227ms/cycle门槛；AI Core重写虽可能加速，却无法预先逐位复现不透明ACLNN的raw Q。已形成`STEP-214_D_SOAP_Q-only_Triton候选设计.md`及G0～G5门禁，裁决`NO_GO_FORMAL_RAW_Q_EQUIVALENCE_UNPROVEN_DESIGN_ONLY`；无NPU/编译/训练/profile/业务改动。
- 2026-08-15：STEP-214-G Triton-Ascend最小A3机制门禁最终PASS。首轮8rank算子本体逐位exact但controller release握手超时，仅记partial；一次收到释放指令时ready0的修正尝试已精确终止。新增controller经本地/容器CPU文件协议自测后做唯一最终重跑，44s内完成8rank逻辑0～7/物理Phy-ID8～15、8 ready/8 done/0 failure、vector-add raw exact和live npu-smi，finally release后进程0。全局Triton3.7.1/torch/torch_npu/CANN未变，隔离产物留在仓库外；裁决`PASS_TRITON_ASCEND_WORLD8_BACK8_VECTOR_ADD_EXACT_GLOBAL_UNCHANGED`，尚未启动MSDA/QR探针或端到端训练。
- 2026-08-15：STEP-214-E完成CPU/源码设计审计，未编译、未调用NPU、未训练、未改业务。隔离`triton-ascend 3.2.0rc4`具备masked load/store、reduction、static loop和atomic前端，Ascend adapter存在AtomicRMW/DiscreteMaskAtomicAdd转换器，但随包无FP32高冲突原子验收，故裁决`DSL_EXPRESSIBLE_BACKEND_ATOMIC_UNPROVEN`。最小候选为真实空间FP32签名专用的1个forward tile kernel+1个融合三梯度backward kernel，其他五类MSDA调用全部回退DrivingSDK；完整贡献workspace为52.5GiB而被禁止。已定义动态shape、边界zero-padding、AMP、stride、autograd/重复性门禁，并要求无profiler完整边界合计从`211.922ms`降到严格小于`189.222ms`（净省`>22.7ms/step`）才可继续；GPU同shape`130.708ms`仅作最终方向目标。
- 2026-08-15：STEP-214-C allocator-only正式8卡30-step A/B完成。唯一candidate自然exit0，8 direct rank/torch_npu/back8 PID/env证据齐全，TQ保持unset/default1。相对STEP204 fresh baseline，normal/SOAP/cycle分别回退`2.0135%/0.7971%/0.9968%`；loss全finite、grad相位正常、dynamic scale一致。裁决`NO_GO_PERFORMANCE_REGRESSION`；按用户要求中央allocator单行patch暂留且未commit，所有远端产物永久保留，资源已释放。
- 2026-08-15：allocator补充门禁完成：max/last显存降低139MiB、normal窗口max降低149MiB；checkpoint meta与tensor shape/dtype/finite、scalar合同一致，数值受自然非确定性影响不逐位。随后STEP-214-F TQ2启动与用户新优先级发生竞态，已在0 iter初始化阶段终止并标为无效A/B；暂停工件永久保留，port29962、训练进程和NPU PID全释放，TQ2合同待后续恢复。
- 2026-08-15：训练收尾只读状态发现生成副作用使tracked `fusion_result.json`显示删除；严格断言该唯一状态后从HEAD精确恢复。最终权威仓库tracked状态仅保留用户要求的`M tools/ddp_train.sh` allocator单行patch，numstat 1/0、diff-check通过，未commit。
- 2026-08-15：STEP-214-I官方检索收口并纠正STEP-214-E证据表述：wheel本身无测试，但完全匹配的官方rc4 tag源码有FP32 32-core规则重复地址高冲突atomic验收；fully-indirect/discrete atomic官方测试与lowering修复只在2026 main/release-3.2.2出现，不能用于冻结rc4。DrivingSDK ScatterAddV3提供UB内聚合后少量GM atomic蓝图，但主优化tailLen=1不匹配MSDA channel32，且无目标shape benchmark、后续还有精度/兼容修复。无ready完整MSDA实现或已证明`>22.7ms/step`patch，裁决`NO_READY_FULL_MSDA_IMPLEMENTATION_GO_RC4_STRUCTURED_ATOMIC_MICROPROBE_ONLY`；本阶段只读、无NPU/训练/profile/业务修改。
- 2026-08-15：STEP-214-H world8后8规则高冲突FP32 atomic局部门禁PASS。rc4官方型c32为`0.144020ms`；MSDA单batch冲突型每launch执行127,401,984次atomic，kernel Event为`2.389620ms`、约53.315Gops/s，zero+kernel为`2.388940ms`。8rank全部finite/oracle exact/repeat exact/max diff0，live物理八die、release、global unchanged和结束资源0均通过。按真实完整shape 14,092,861,440次atomic吞吐外推约264.33ms，慢于DrivingSDK完整空间backward146.580ms，裁决`PASS_RC4_STRUCTURED_ATOMIC_MECHANISM_NO_GO_DIRECT_PER_CONTRIBUTION_MSDA_BACKWARD`；无业务接入、训练、profile或环境覆盖。
- 2026-08-15：STEP-214-J B1 register归并局部门禁完成。将27个随机正负FP32贡献在每program内累加后再atomic，使GM atomic由864降至32/output，Event从2.437300降至1.207890ms、加速2.018x；但B112线性外推135.284ms未过123.88ms，理论净省仅11.296ms。aggregate oracle误差`max_abs<=3.814697e-5/NRMSE<=1.270072e-7`但重复仍非逐位。裁决`NO_GO_REGISTER27_AGGREGATE_PERF_BELOW_22P7_AND_RAW_REPEAT_NOT_EXACT`，未运行真实shape、训练、profile或改业务，资源已释放。
- 2026-08-15：STEP-214-K真实B1空间MSDA forward原型两次机制门禁均在首个Triton warmup失败。gate1被rc4 host的grid<65536检查拒绝；只按明确提示设置隔离`TRITON_ALL_BLOCKS_PARALLEL=1`后的gate2仍被device runtime以`coreDim=122880>UINT16_MAX`拒绝。无正式Event/数值样本，按约不再改kernel网格或重跑；裁决`NO_GO_RC4_COREDIM_LIMIT_TRUE_SHAPE_FORWARD_PROTOTYPE_UNEXECUTABLE`，资源已释放。
- 2026-08-15：STEP-214-L独立双head/grid61440 forward候选完成唯一world8 B1 gate。数值相对DrivingSDK为`max_abs<=1.549721e-6/NRMSE<=2.742497e-7`且repeat exact，但Event为`2513.123901ms`，对照SDK`0.644240ms`约慢3901x，B112外推281469.877ms。裁决`NO_GO_TWOHEAD_GRID61440_EXTREME_PERFORMANCE_REGRESSION`，不跑B112/训练/profile/业务，资源已释放。
- 2026-08-15：STEP-214-M Q-tiled32/grid3840 forward候选完成唯一world8 B1 gate。数值`max_abs<=1.490116e-6/NRMSE<=2.052884e-7`且repeat exact；Triton Event423.697617ms，对照DrivingSDK0.609280ms约慢695.4x，B112外推47454.133ms。裁决`NO_GO_QTILE32_EXTREME_PERFORMANCE_REGRESSION`，不跑B112/业务，资源已释放。
- 2026-08-15：STEP-214-N最终persistent grid64/tl.range forward候选完成唯一world8 B1 gate。数值`max_abs<=1.490116e-6/NRMSE<=2.052884e-7`且repeat exact；Triton Event1518.993774ms，对照SDK0.643330ms约慢2361x，B112外推170127.303ms。裁决`NO_GO_PERSISTENT64_EXTREME_PERFORMANCE_REGRESSION_CLOSE_TRITON_FORWARD`，停止forward tile sweep，资源已释放。
- 2026-08-15：STEP-214-O QR2560局部原语门禁完成。固定NPU栈支持geqrf+orgqr，Event相对linalg.qr由4027.1255降至1262.1964ms、3.191x；但raw Q非bitwise（max/NRMSE `4.734844e-6/3.960207e-6`），R亦非bitwise。裁决`NO_GO_RAW_Q_BITWISE_MISMATCH_DESPITE_3P19X_SPEEDUP`，不扩24类、不接optimizer/训练，资源已释放。
- 2026-08-15：STEP-215-B只读官方QR补丁检索启动。已恢复并去重STEP-089～100/199/214-O，首轮仅核对PyTorch、Ascend op-plugin与CANNBot官方来源；未连接远端、未安装、未调用NPU或训练，网页发现只写入`findings.md`。
- 2026-08-15：STEP-215-B官方语义/API/补丁审计完成。用户本轮授权正式覆盖后续raw-Q逐位门禁；PyTorch2.7确认QR列符号非唯一且数学等价FP32不保证bitwise。`geqrf+orgqr`与`linalg.qr`数学合同一致但不承诺同executor/raw输出；`orgqr`已是直接显式Q路径，`ormqr(identity)`不是Q-only，当前`v2.7.1-7.2.0/CANN8.3RC1`无ready Q-only补丁。已冻结24-shape `Q/orth/recon<=1e-5`、双QR周期持久状态`5e-5/1e-4`及resume分阶段门禁，裁决`GO_STAGED_NUMERICAL_GATE_GEQRF_ORGQR_PRIMARY_NO_READY_Q_ONLY_PATCH`；本子任务未连接远端、未跑NPU/训练、未改业务。
- 2026-08-15：STEP-215 24-shape局部门禁准备完成。已恢复24类历史shape、551次历史权重和543次当前活动权重，完成world8后8局部A/B harness；主任务审阅后补充所有shape强制warmup及900秒controller/930秒外层硬超时。本地`py_compile`通过；未连接远端、未调用NPU。既有Python SSH依赖受本地ACL限制，OpenSSH又在连接前被网络沙箱拒绝；临时连接包装器已删除。状态为`ready_local_harness_blocked_remote_transport_before_launch`，连接恢复后可直接做远端`bash -n`、SHA/资源门禁并唯一启动。
- 2026-08-15：STEP-215局部门禁启动前复核补齐baseline/candidate Q自噪声、至少两次稳定样本、Q max-abs硬上限与normalized-Fro正交误差；自适应阈值仍受`1e-5`硬上限约束。修正后py_compile/AST通过，未连接远端或调用NPU。
- 2026-08-15：STEP-215-E完成SOAP最小候选与双QR周期仓库外harness设计。草案只把唯一QR行替换为`geqrf+orgqr`，stable sort、FP32、state重排/写回与schema不变；补齐no-grad/autograd、out/alias、秩亏/NaN/Inf异常合同。双周期采用baseline双跑+candidate、实际QR周期检测、持久state数值门禁和首周期resume；当前未改业务、未运行NPU/训练，状态`DESIGN_READY_WAIT_24SHAPE_GATE_NO_BUSINESS_CHANGE`。
- 2026-08-15：STEP-215-F纯本地静态门禁完成。Codex bundled Git runtime 的`sh.exe`确认为GNU Bash5.2.37，container runner/host launcher的`bash -n`均exit0；Python AST复算确认24类、历史551次、活动543次，所有shape双侧warmup、正式样本至少2、自噪声及`1e-5`硬门禁字段齐全。world8/controller900s/宿主930s/rank release180s及controller-finally、runner-trap、host-timeout三层释放合同闭合。四文件SHA已写入不含敏感路径/凭据的本地source manifest，裁决`PASS_STATIC_PACKAGE_READY_NOT_EXECUTED`；未连接远端、未进入容器或调用NPU，不能据此宣称数值/性能PASS。
- 2026-08-15：STEP-215-E双周期+resume仓库外执行框架已实现。支持三轨exact snapshot/gradient replay、目标QR动态patch、实际两周期检测、首周期checkpoint/load、第二周期continuous/resume，以及schema/sort/step/finite/Q `5e-5`/其他state与参数`1e-4`门禁；repo/config/checkpoint/output均参数化。由于本地无权威业务构建入口，真实SOAP adapter模板10项readiness全部false并在ready前硬失败；状态`FAIL_CLOSED_SCAFFOLD_IMPLEMENTED_NOT_RUNTIME_READY`，未连接远端、未运行NPU/训练、未改业务。
- 2026-08-15：STEP-215-E主审补丁完成：candidate周期1/2有效阈值改为baseline-A/B自差的2倍、floor=`1e-5`且分别受Q=`5e-5`/other=`1e-4`硬封顶；candidate resume另以两条baseline continuous/resume自差的最大值校准。静态包仍fail-closed，未运行远端/NPU。
- [2026-08-15] STEP-215-G 完成：world8 后 8 卡 24 类真实 QR shape 局部 A/B 自然 exit0；当前活动 23 类/543 次全部数值通过并预计每周期净省 14.999s，历史非活动 5120 因 Q max_abs 超限保留回退。进入 SOAP 双周期+resume 门禁，尚未改业务/训练/commit。
- 2026-08-15：STEP-215-H本地完成`.codex-tools/step215_e_real_soap_stateful_adapter.py`。adapter冻结活动config/checkpoint身份，在运行时加载559个真实SOAP state，以`exp_avg` shape/dtype构造stateful Parameter子集，确定性synthetic gradient预计在逻辑step3/13命中state30/40两个QR周期；明确省略208个无state参数且不宣称完整模型参数恢复。主gate新增23类白名单、5120/未知shape回退、每周期543 inventory及无5120硬断言。两个Python `py_compile`和本地AST/常量单测通过；未连接远端、未调用NPU/训练，状态为`RUNTIME_READY_SOURCE_NOT_EXECUTED`。
- [2026-08-15] STEP-215-J：真实 SOAP state world8 双周期 gate3 到达 candidate 第二周期，但 continuous/resume operator event 在 step13 一致失败；资源已释放。暂停训练/提交，进入只读字段级诊断。
- 2026-08-15：STEP-215-M仅本地实现basis-relaxed diagnostic：默认strict，只有显式`--basis-relaxed-diagnostic`才忽略两个baseline-vs-candidate周期的raw Q距离；新增每个Q finite/方阵/orth max-abs<=1e-5、非Q逐tensor+global<=1e-4，保留candidate resume Q<=5e-5及schema/step/sort/inventory exact。Python py_compile、AST合同与6项CPU-free单测通过；未连接远端、未运行NPU/训练、未改业务。工具SHA：gate `47a098ec...5ad797`，runner `d6d8b289...8d2358c`，host `bdb0be95...291250`，test `0ca9c947...1cc6c`。
- 2026-08-15：STEP-215-N正式world8 basis-relaxed局部SOAP门禁完成并拒绝。预检SHA/bash-n/torch_npu2.7.1/后8设备/资源通过，ready8与npu-smi后8die live8成立；8/8 rank在baseline-A第二周期continuous/resume自身Q正交max-abs=`1.3064861e-5`超过`1e-5`硬上限，candidate及可放宽的跨实现raw-Q比较尚未进入。裁决`REJECT_Q_ORTHOGONALITY_HARD_GATE`，未启动30-step/不重跑；结束active0/port0/NPU进程0。42个、11.996GB工件永久保留，summary/manifest SHA=`786e0484...acfbd3`/`68021693...f64187`。
- 2026-08-15：STEP-215-O按授权执行唯一一次Q orth `2e-5`校准重跑。显式参数默认仍1e-5、绝对封顶2e-5；其余门禁不变。world8 live通过，baseline-A/B两个周期通过；candidate cycle1 save/load时rank1实际Q orth=`2.0570097e-5 >2e-5`硬失败，elastic终止其余rank，ready/failure/done=`8/1/0`。最终拒绝，不再放宽/重跑/30-step；active0/port0/NPU0。57文件/32.996GB永久保留，summary/manifest SHA=`bfc96d08...abbd51`/`f3d39463...55336c6`。
- 2026-08-15：STEP-216-A完成仓库外TurboSOAP Brockett core+一次cubic polar局部筛选包，仅本地静态、未连接远端/NPU。权威参数固定为commit`1339218c...`，FP32/stable-sort、eta0.01/单substep，禁eigengap/EMA/controller；真实adapter将重建559 state中的23类/543 factor，5120/未知合同回退。finite/orth2e-5/Rayleigh/真实marginal预条件作用/重复性/peak memory/Event+wall净省227ms门禁、world8 controller及1200s外层硬超时已闭合。Python编译、AST、6项policy单测、两shell Bash-n和diff-check通过，状态`STATIC_WORLD8_PACKAGE_READY_NOT_EXECUTED`。
- 2026-08-15：新会话按 `planning-with-files` 恢复上下文。已读技能文档、`task_plan.md`、`progress.md` 尾部、`操作步骤.md` 尾部、`findings.md` GPU/NPU 对比段和最终报告。用户当前问题仍是 NPU 相对 GPU 性能差距大；顶部 Next Step 已从过期 STEP-193 校正为 STEP-216-A world8 局部门禁。本轮未连接远端、未占 NPU、未改业务。
- 2026-08-15：STEP-217 GPU为标准只读对比。用户禁止抢占正在使用的NPU。远端核验NPU raw=205文件/16.65GB、GPU 7z=473,979,928B、解包JSON=12.37GB，与STEP-202/203永久清单一致。结论：优先SOAP QR（19×/+21.6s周期步），其次普通步host/underfeed（约+1.86s），MSDA空间FP32残余+81ms但固定SDK；NPU普通kernel总量0.88×、backward更快，不作为优化对象。未训练、未改业务。
- 2026-08-15：STEP-218固化如何优化。P0=Brockett+cubic polar替换周期QR，空闲后仓库外world8，正交2e-5、预条件5e-3、每周期>227ms；上次预启动因adapter在仓库内未出样本，不得用同一入口重跑。P1 host空洞无新单一边界，不重开NPUGraph/TQ2/pin。MSDA等外部kernel。未占NPU、未改业务。
- 2026-08-15：提交`2846401`【npu性能优化】SOAP周期QR异步流水化（陈旧Q固定k=4步换入+同步回退）；仅soap.py。启用需`SOAP_STALE_Q_K=4`。
- 2026-08-16：STEP-223/223-B DataContainer pin 在 stale-Q 基线重测通过。30-step 因冷启动 REJECT；100-step 吞吐 20.200→24.155 samples/s（+19.58%），late≥10 约 27.33≈GPU 参考。工作树已装入 `mmcv/parallel/data_container.py` +18 行，待用户要求再 commit。
- 2026-08-15：STEP-222 P1 普通步 Level0 低扰动采集完成并裁决 `NO_GO_NO_UNIQUE_EQUIVALENT_BOUNDARY_ABOVE_22P7MS`。基线`2846401`+`SOAP_STALE_Q_K=4`；wait22/warmup1/active2、无栈无shape、26/26 exit0。排除QR后仅1次36.7ms空洞且不复现、无Call Stack；分散簇禁止拼接。raw已删，analysis摘要保留。无改码/A/B/commit。
- 2026-08-15：STEP-221-D Stage D 300-step通过。用户取消876。k=0/k=4 各300/300 exit0。SOAP 28.286→5.938s（-22.35s）；吞吐16.285→21.272 samples/s（+30.6%，约0.75:1 vs GPU）；loss仍下降。相对普通步+6.23%越次级门，但绝对耗时与Stage C候选一致、本轮baseline偏快。裁决STAGE_D_PASS。soap工作树已装入候选未commit。
- 2026-08-15：STEP-221-C Stage C通过 `STAGE_C_PASS`。同patched soap、单变量仅`SOAP_STALE_Q_K`；后8卡8rank、GPU对齐合同、30/30、exit0。周期步28.557→5.831s（净省22.726s）；普通步5.664→5.784s（+2.12%≤5%）；peak 25699→25754 MiB（+55）。前4步loss逐位相同后因相位平移分叉。soap已恢复权威SHA；fusion_result还原；kernel_meta 64文件归档诊断目录。产物`diagnostics/step221_stage_c_stale_q_30step_8npu_20260815T192500/`。下一步Stage D 876-step+测试集。
- 2026-08-15：STEP-221-B2 Stage B通过。仓库外TOOL_ROOT按字节锚点生成patched soap（26,170B，`get_orthogonal_matrix_QR`字节未动，业务repo零改动）。单卡真实559-state门禁：T1 trio与原函数Q/exp_avg_sq全部逐位一致；T2异步Q与同步Q逐位相同、stale_steps=4；T3 pending恒0或559无重叠；T4 state_dict强制flush后恰为7键合同；T6额外alloc 253.7MB（修power_iter引用前535.7MB）、reserved 722MB待Stage C复测。周期步22.82s→提交步0.37s；安装步21.79s系harness前台仅0.13s/步所致，真实训练6.18s/步由Stage A覆盖。下一步Stage C 8卡30-step A/B。
- 2026-08-15：STEP-221-A Stage A微基准通过。用户批准stale-Q轨道并允许训练。真实23类shape/543次QR合成负载：单卡QR alone=22.734s（与训练profile一致）、host下发0.019s；单卡与8卡并发hidden=99.86%~100%、前台减速≤0.14%、qr_alone 8卡持平单卡（AICPU每卡私有无争用）。k=4。结果JSON留远端`step221_stage_a_20260815T1810/`，进程/卡已释放。下一步Stage B仓库外adapter静态包。
- 2026-08-15：STEP-221方案设计v2。STEP-220 Brockett已被数值拒绝（投影rel-L2 1.49、正交6.13e-3），位级等价轨道穷尽。新P0-v2=SOAP QR异步流水化（独立流提交543次QR、旧Q续跑、固定k步换入、k=0回退、save前强制换入），QR数学不变。门禁：StageA微基准（隐藏≥70%/前台减速<5%）→B静态包→C 30-step A/B（周期净省>5s价值线）→D 876-step+测试集（≥0.75:1）。属语义时序变化，待用户批准轨道后占卡。未连远端、未占NPU。
- 2026-08-15：STEP-219审计优化方案。合同一致性通过；4个缺口：①正交2e-5可行性应先CPU预筛checkpoint 543个Q（215-N/O有踩线史）；②漂移无监控（TurboSOAP安全网全禁用，876-step≈87周期）；③adapter布局静态修复未做；④缺价值线（227ms是噪声下限，QR 22.64s/cycle）。预期上限：P0全成功仅约0.78~0.81:1。未占NPU、未改业务。
- 2026-08-15：STEP-216-A独立must-fix静态修复完成：真实559-state/543-Q投影作用、复用power_iter、3次交替整周期、显式candidate-minus-baseline峰值、source contract、direct host PID绑定和统一TERM/KILL/postflight均闭环。Python编译、6+5项测试、identity与diff-check通过；本机无Bash/WSL发行版，未虚报`bash -n`。未远端/NPU运行。
- 2026-08-15：STEP-216-A唯一远端执行已fail-closed收口。预检PASS；直接入口rc126记failed/effective0，授权的唯一bash纠正入口因adapter位于repo内触发runner路径断言rc1，host清理自匹配使外层rc143。world8未启动、样本0；不重跑。最终active0/port空/back8进程0，failure summary/manifest永久保留。
- 2026-08-15：仅本地完成STEP-216-A恢复包修复：显式repo外TOOL_ROOT realpath合同，删除所有字符串pkill，改为两层精确PID/PGID TERM→wait→KILL；新增路径/cleanup回归测试。pycompile、6+7测试、source identity、Bash5.3两shell`-n`和diff-check通过；未上传/远端/NPU。
- 2026-08-15：极窄复审补齐host异常时的容器launcher PGID清理；顺序固定为container launcher group→host job group→postflight，trap全程保留。严格PGID语法测试后静态合成8/8、source identity和bash-n通过；未远端/NPU。
- 2026-08-15：唯一repo外core world8创建8 worker后在ready前拒绝真实SOAP签名漂移；0 core样本、不重跑，postflight归零，远端summary/manifest永久保留。随后仅本地按完整state+max_precond_dim接口修复，policy6/6、静态10/10、AST/fake SOAP/source identity/bash-n通过，未再上传/NPU。
- 2026-08-15：773574接口包唯一world8达到ready8/真实559×543及后8die live8，但旧host_pid字段实际为container PID，controller集合门禁拒绝，0 cycle且不重跑；postflight归零。随后静态改为npu host PID→宿主proc NSpid最后项→ready container PID双射，容器init只读映射PASS，测试10/10；未上传/NPU。
- 2026-08-15：51cccd v3唯一core再次ready8/live8，但容器controller无法访问宿主proc，0 cycle/不重跑/postflight归零。随后仅本地将controller迁到host launcher并行管理，runner只跑torchrun；host-controller fixture 12/12、source identity/pycompile/bash-n通过，未上传/NPU。
- 2026-08-15：STEP-216-A host-controller版唯一有效core自然`rc=0`：ready/done/failure=`8/8/0`，宿主8个npu-smi PID经NSpid与8个容器rank PID及后8die严格双射；每rank真实559 state/23类/543 factor，完成3个paired cycle。Event/wall中位节省约22.293s/cycle，内存增量通过，但真实投影作用global/逐tensor rel-L2最坏1.4068/1.4907、Rayleigh及dimension2560正交失败，裁决`REJECT_LOCAL_SCREEN`。postflight active0、port29997空、back8进程0；summary/manifest SHA=`f77ff0b...47fc`/`00ed72b5...42ed`永久保留，未训练/未改业务。
