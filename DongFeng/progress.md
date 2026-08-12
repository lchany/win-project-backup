# Progress Log

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
| Where am I? | Phase 1：只读审计与计划制定 |
| Where am I going? | 基线、随机性移除、算子优化、系统优化、回归交付 |
| What's the goal? | 在保持正确性的前提下获得可复现的 Ascend NPU 性能收益 |
| What have I learned? | 见 `findings.md` |
| What have I done? | 已建立规划文件与批准门禁，尚未修改业务代码 |
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
