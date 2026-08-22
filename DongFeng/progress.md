# Progress Log

## Session: 2026-08-22 MX QrV2 实现方案固定与审核

- 已对现有 `MX_QrV2_完整修复实施方案.md` 进行执行前独立判断：历史第4～5节仍容易被误解为当前发布方案。
- 已新增第17节作为当前唯一执行入口：区分已确认事实、合理推测和未验证假设，禁止直接发布 v1～v5 或 STEP375～377 诊断产物。
- 已固定“单次诊断→唯一最小 release delta→静态/构建→CPU语义→后8卡首输入→状态化→30-step loss→耗时”的顺序和停止条件。
- 本轮只修订规划文档，未修改原 ZIP/算子源码/installed package，未连接远端，未启动 NPU 或训练。
- 两路独立子 Agent 正在审核源码证据边界和验收门禁；审核问题关闭前不进入实现。
- 两路审核首轮发现 v5 历史发布文本与当前方案冲突、CPU oracle 顺序/dtype 不完整、顶层旧 npu2 路径、性能统计口径及单 delta 假设等问题；均已修订。
- 最终方案允许“最小必要 delta 集”，但诊断每轮仍只改一个原子 delta，发布前须逐项消融证明每项必要。
- 最终复审裁决：P0=0、P1=0；非阻断 P2 已进一步固定 bootstrap seed/重采样单位/中心统计量、CI上界+点估计+最差批次同时裁决，以及温度/host load 仅作观察。
- 方案审核通过，但修复尚未实施或验收；下一阶段仍须先审核 STEP377 phase-transition，且在性能测试前由用户锁定 fixed/original MX 最大耗时回归。
- attempt6 armed-state单测发生production backend逃逸，已立即回退`NPU_READY=False`并永久废弃attempt6。原位审计证明8/8 rank在QrV2之前因installed custom OPP硬编码路径不存在而退出；全部NPU/exact case已空，未得到算子证据。
- attempt7已保持默认未武装；controller test新增production backend全局禁止且28/28 PASS。当前并行修复真实installed root传递与pre-rank-evidence退出清场。
- attempt7修正后已达到8/8 ready，但因`NSpid` tab分隔不被解析而在release gate前失败，未加载输入/调用QrV2。已回退未武装attempt8；外层timeout后exact case/端口/后8卡均已归零。

历史全文已备份到 `planning_backup/2026-08-18/`。本文件从 2026-08-18 规划文件轮换后重新开始。

## Session: 2026-08-21 本地相关文件读取

- 用户明确调用 `planning-with-files`，已完整读取技能说明并恢复现有三份规划文件。
- 已盘点根目录 Markdown：`操作步骤.md` 7154 行，另有项目状态、性能报告、QR 与分阶段报告等文件。
- 已脱敏读取 `机器IP.md`，确认连接链路及主训练机末段 `42`；未远程连接。
- 已派发独立 GBrain bootstrap worker；因当前环境缺少 `ensure_project` 能力且本地 CLI 未配置 brain，初始化未完成。
- 当前正在读取项目规则、外部权威主机映射和操作记录中的相关段落。
- 已完整读取项目 `AGENTS.md`，并脱敏读取 `/home/l30002999/import-md/hw-import-ip.md`；未输出地址、账号或密码原值。
- 用户随后明确禁止读取当前目录外文件；已立即收紧范围，后续仅访问项目目录内文件。
- 已建立顶层 Markdown 标题索引，并读取 `PROJECT_STATUS.md`、`DrivingSDK优化研究与实施计划.md`、`QR算子.md` 的相关内容；识别出历史状态与当前状态的时间差。
- 用户进一步澄清：不得读取其他目录中的 `planning-with-files` 技能材料；后续严格只使用当前目录文件。
- 已读取当前三份规划文件、项目规则、脱敏主机说明、状态/研究计划/QR 文档、STEP-208/226/227 验收材料，以及 `操作步骤.md` STEP-315～335。
- 已完成新旧口径归并：以 STEP-334 为最新 QR 定位结论；旧 STEP-227 的接近 1:1 与当前双门禁合同不可直接等同；Phase 13 完成。

## Session: 2026-08-21 MX QR 根因验证

- 用户将当前目标明确为验证所提供 MX QR 算子出现问题的原因。
- 已只读检查 STEP-319～323 与当前 `soap.py`；发现既有 raw Q/R 对照 oracle 不足，且 isolated 测试未重放训练调用序列和 SOAP 下游状态。
- 当前先区分算子数学合同故障与 SOAP 集成/基连续性故障；尚未连接远端或修改训练代码。
- 已审计 STEP-315 hook 与本地 `qr_v2.cpp`：确认 dump 只保存输入且引入同步扰动；确认最后 block 存在释放后继续读取 `tLocal/vLocal` 的静态路径，并伴随零列 tiling 未初始化风险。
- 已在本地重新读取 STEP-260 的 8 份真实 A/Q/R：所有 rank 都是有限 A，但 Q 的最后 64 整列和 R 上三角最后 64 列全部非有限，动态失败边界与最后 tile 精确一致。
- 用户新增硬门禁：修复后逐 step loss 相对 GPU log 必须全部 ≤2%；耗时允许偏差尚待明确，未明确前不作最终通过裁决。
- STEP-337 只读 provenance 通过：远端 wrapper/qr_v2.cpp 与本地快照哈希一致，实际编译 `.o` 已定位；未启动算子、未改远端。

## Session: 2026-08-18

### Phase 8: SOAP 精度合同与社区 QR
- **Status:** in_progress
- **Started:** 2026-08-18
- Actions taken:
  - 备份根目录 `task_plan.md` / `findings.md` / `progress.md` 到 `planning_backup/2026-08-18/`。
  - 按当前 STEP-265 状态重建精简规划文件，去掉 STEP-001～264 历史正文。
  - STEP-268：后 8 卡对 192×192 加严测试 53 次。47 通过；同一份 BAD A 在部分卡上 AICore 崩溃（QrV2 MTE 越界）。未改业务代码。
  - STEP-269：29 例隔离扫描完成；layout/warmup 非触发条件；npu2–7 507015。已读 `qr_v2.cpp` 最后 tile 空 LARFB 路径。
  - STEP-270：npu2 上除 64×64（AICPU 回退）外全部 507015；钉死 QrV2 在 visible npu2–7 设备分域失效。
  - STEP-272：前 8 卡空闲后 `eye(192)` 探针 7/8 失败；换卡不能规避。失败跟 visible npu 下标 2–7 有关，不是 phy 10–15 特有。
  - STEP-273：bypass 前后 8 卡 16/16。
  - STEP-274：后 8 卡 + bypass 完整 30 step，rc=0，无 NaN；loss **30/30 ≤2%**、23/30 ≤1%；Iter2–30 372.9 s（2.67×GPU）。
  - STEP-282：回滚 site-packages 192 bypass 与 SOAP 诊断改写；`soap.py` 相对 `669a138` 只换 `mx_driving_cloud.linalg.qr`。未 commit。
  - STEP-283：删除 git 仓库内未跟踪测试/诊断残留（`$REPO/diagnostics/`、`kernel_meta/`、`no_track_*.pt.trace.7z`）；共享盘 `.../wfc1_leicheng/diagnostics` 未动。SOAP dump 标记为 0。
  - STEP-284：只提交 `soap.py` 为 `9565044`。相对 `669a138` 仅 5 行 QR 替换。未 push。
  - STEP-285：8 个 BAD `.pt` × 8 可见卡独立调用官方 `mx_driving_cloud.linalg.qr`。npu0 8/8 有限；npu2–7 48/48 崩 507015。
  - STEP-294：按用户要求准备在新 NPU 机用同一 BAD tensor 对 logical npu1–7 做独立单卡 QR 测试；前检发现完整名称 `mapqr-leicheng` 的容器不存在（运行/停止列表均无），物理卡 4–15 正有高 AICore 占用。遵守容器与资源门禁，未启动算子调用，未改远端环境。
  - STEP-299：按最新 `机器IP.md` 连接 42 远端，在精确容器 `mapqr-leicheng` 中上传本地 rank0 BAD tensor 副本；后 7 张物理卡分别单独可见、各独立进程调用一次官方 `mx_driving_cloud.linalg.qr`。7/7 Q/R 有限，NaN/Inf=0，无 507015；输入 SHA-256 与本地一致，测试结束无残留进程。
  - STEP-301：后 8 卡同时可见，logical npu1–7 各独立进程显式 `torch.npu.set_device(k)` 后调用同一 BAD tensor。7/7 Q/R 有限、NaN/Inf=0、无 507015，数值与单卡可见结果一致。由此纠正 STEP-270/285 的“设备分域”归因：旧 harness 只把 A 放到 npu:k，却未设置 current device。
  - STEP-302：修正给算子同事的 README 与 `repro_qr.py`，加入强制 `set_device`/current-device 断言并撤销“npu2–7 必崩”口径；生成不覆盖旧包的新文件 `qr_operator_repro_for_colleague_step301_corrected.zip`。
  - STEP-303：在 42 远端 `mapqr-leicheng` 容器内，保持 8 卡同时可见并显式 `set_device`，对同一 BAD192 在 logical npu0/1/2 各连续调用 512 次官方 QR。3/3 跑满，Q/R 全有限、无 NaN/Inf、无 507015，`recon_absmax≈1.954e-14`；说明当前单算子高频重放仍未复现训练态 NaN。
- Files created/modified:
  - `planning_backup/2026-08-18/task_plan.md`（备份，167380 bytes）
  - `planning_backup/2026-08-18/findings.md`（备份，376500 bytes）
  - `planning_backup/2026-08-18/progress.md`（备份，189700 bytes）
  - 根目录三份规划文件（重建）
  - `操作步骤.md`（追加 STEP-266、STEP-274）

## Carry-forward Snapshot
- STEP-265 已完成根因拆分；STEP-266 已把 NaN 钉到 mx QR 最后 64 列 tile。未改业务代码、未新开训练。
- 8 个 BAD `.pt` 已在本地并传到同事机 `/home/ubuntu/`。
- 权威 HEAD `9565044`：`669a138` SOAP + 官方 mx QR。未 push、未开训。
- 双门禁下精度达标路径现有两条：63861df CPU FP64 SOAP（30/30 ≤2%，6.2–6.4×GPU）；STEP-274 当前工作树 + QR bypass（30/30 ≤2%，2.67×GPU）。后者仍依赖容器内 `linalg.py` patch，未 commit。

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| STEP-265 根因拆分 | 既有 30-step A/B + STEP-260 dump | 区分 NaN 与 11/30 | NaN=社区 QR；11/30=63861df 风格 SOAP | ✓ |
| BAD tensor 直传 | 8× `[192,192]` `.pt` | 同事机可 `torch.load` | 8/8 上传成功 | ✓ |
| STEP-269 布局/warmup | 9 layout + warmup + 8卡 | 排除 layout/warmup | npu0 23/23 OK；npu2–7 6×507015 | ✓ |
| STEP-270 设备分域 | identity/sample/BAD×8卡 | 定位 QrV2 设备 bug | npu2–7 任意>80 崩；64×64 AICPU OK | ✓ |
| STEP-272 前8卡 eye(192) | 物理 0–7 空闲探针 | 判断换卡能否规避 | npu0 OK；npu1 recon=1；npu2–7 507015 | ✓ |
| STEP-274 bypass 30step | 后8卡 + MX_QR_VALIDATION_BYPASS=1 | 对比 GPU loss/耗时 | 30/30 ≤2%，372.9 s / 2.67×GPU | ✓ |
| STEP-275 commit+800step | 后8卡 + 硬编码 linalg@10f897d | 800 step 训练 | Iter13 假包遮蔽官方 wheel 失败 | ✗ |
| STEP-285 8×BAD 官方 QR 单测 | 8 文件 × 8 可见卡，无 bypass | npu0 全过；npu2–7 全 507015 | 64 例：16 有限 / 48 崩溃 | ✓ |
| STEP-294 新机 BAD 单卡复测前检 | logical npu1–7，各独立进程 | `mapqr-leicheng` 存在且目标卡可用 | 同名容器不存在；物理卡 4–15 忙 | blocked |
| STEP-299 7 张物理卡单独可见 QR | 同一 rank0 BAD `[192,192]`，每进程仅 1 张后卡可见 | 判断 NaN/Inf/507015 是否随物理卡单卡必现 | 7/7 有限；NaN/Inf=0；507015=0；recon_absmax=1.954e-14 | ✓ |
| STEP-301 8 卡可见 + 显式 set_device | 同一 BAD，logical npu1–7 各独立进程 | 验证旧 507015 是否由 current-device 漏绑导致 | 7/7 有限；NaN/Inf=0；507015=0；current_device 全匹配 | ✓ |
| STEP-303 高频 BAD192 QR 重放 | 同一 BAD `[192,192]`，8 卡可见，logical npu0/1/2 各 512 次 | 检查正确 `set_device` 下是否仍会在重复调用中出 NaN/507015 | 3/3 跑满；Q/R 全有限；NaN/Inf=0；507015=0；recon_absmax=1.954e-14 | ✓ |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-08-18 | 同事机账号 `ubantu` 认证失败 | 1 | 改用 `ubuntu` |
| 2026-08-18 | 跳板机直连同事机 Timeout | 1 | 本机下载后再直传 |
| 2026-08-19 | STEP-299 启动器等待 SSH 输出 30 秒超时 | 1 | 未重复启动；只读轮询既有远端目录，确认任务正常完成 rc=0 |
| 2026-08-19 | STEP-303 首次 SSH 开 channel 超时；二次运行 heredoc 包装尾部报 `NameError: PY` | 各 1 | 按规则只做一次受控重试；第二次异常发生在 summary 已写出之后，结果有效，仅需后续清理本地包装脚本 |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 8 / STEP-285：8 个 BAD `.pt` 官方 QR 单测完成 |
| Where am I going? | 同事侧需用 8 可见卡测 npu2；禁止只用 npu0 判通过 |
| What's the goal? | 8 NPU 吞吐 1:1，且 loss ≤2%、耗时大幅下降 |
| What have I learned? | 同事「单测没问题」= npu0 冷跑；同 A 在 npu2–7 必崩 507015 |
| What have I done? | 提交 `9565044`；64 例独立 QR 单测 |

---
*Update after completing each phase or encountering errors*

## Session: 2026-08-21 STEP358 修复版发布门禁

- 完整 shadow、8-rank concrete AIC、公开输入/内部 padding/调用账本、80 个确定性矩阵族、default/dedicated stream 状态序列和 owned-PID 清场已实现。
- 第二轮审核发现的 partial-ready 清理、顶层 Torch 提前导入、OPP重复路径、profile计数、输入SHA与矩阵族覆盖问题已修正。
- 本地 `py_compile`、构建器17项测试、oracle20项测试和 diff 检查通过；等待第三轮只读终审，尚未运行远端 STEP358。

## Session: 2026-08-19 GitHub 发布

- 启用 `github:yeet` 与 `planning-with-files` 工作流。
- GBrain 只读经验召回未命中相关经验；无正式 worker token，未生成 receipt。
- 确认仓库根目录、`main` 分支和 GitHub 远端；当前项目存在大量修改及未跟踪文件。
- 安装 GitHub CLI 2.97.0；认证检查显示尚未登录。
- 发现 `DongFeng` 内有约 452 MB 原始 profiling 压缩包，兄弟目录另有约 711 MB 安装包；二者均不能作为普通 GitHub 文件提交，本次仅处理 `DongFeng`。
- 发现 `机器IP.md` 有本地修改；按项目规则明确排除，不把连接信息写入日志或提交。
- 根据用户确认，统一排除大量二进制、缓存、训练与 profiling 产物；发布分支为 `codex/publish-local-project`，暂存区仅含代码、脚本、配置、补丁与报告，未发现常见二进制扩展。
- Gitleaks 安装成功；首次执行因 PowerShell 对不同对象类型的可执行路径属性解析错误而未启动扫描，已记录并改用类型兼容路径解析。
- Gitleaks 已实际扫描约 5.01 MB 暂存内容，发现 8 个疑似敏感项；未输出或记录匹配值，进入按文件/行号/规则定位阶段。
- 已定位疑似项分布在两个 `train_spetr.py` 副本和一份生成的 HTML 报告；首次只提取关键词元数据的 PowerShell 命令有语法错误，未读取或输出敏感值，已改写实现。
- 8 个命中均对应硬编码 `wandb_key`。第一次按引号结构替换时因原行格式不同而未匹配，未修改文件；改用变量名整行锚定的固定空值替换。
- 第二次同类替换仍未匹配，已停止重试并派发只读故障召回。根因初步确认是 PowerShell 单引号正则里多写了一层反斜杠，导致匹配的是字面 `\s` 而非空白字符类；未读取、输出或改写秘密值。
- 故障召回分类为“部分适用”：只按元数据定位、停止盲替换、修改后逐文件复核并重扫。结构化替换脚本首次运行在错误消息插值处发生 PowerShell 解析错误，早于文件读写；已明确变量边界修正。
- 已精确移除 8 处硬编码 `wandb_key`，未输出原值；Gitleaks 复扫约 5.01 MB 暂存内容为 0 命中。
- 暂存区 267 个文件、常见二进制候选 0；忽略清单覆盖 1 个原始 trace 压缩包、训练日志、16 个 tensor 和 539 个 Python 缓存文件。
- 149 个暂存 Python 文件全部通过 AST 语法解析。
- `git diff --cached --check` 仅报告历史源快照、补丁与生成 HTML 的尾随空白/EOF 空行；为避免改变补丁语义不做全量格式化。
- GitHub CLI 仍未登录；GitHub 连接器首次元数据查询发生传输错误，尚未产生任何远端写入。
- 最终门禁：267 个暂存文件全部位于 `DongFeng/`，二进制候选 0，最大文件 1.155 MB，Gitleaks 0 命中。首次提交因仓库未配置作者身份而停止，未生成 commit；将复用上一条提交作者并仅做 repo-local 配置。
- 已复用上一条提交作者做 repo-local Git 配置，生成提交 `0fa5e1b`（归档本地优化脚本与报告）。
- 首批提交后检测到 3 个新出现的 STEP-306 源码脚本；作为独立功能增量继续验证和提交，不纳入凭据、缓存或 GBrain 本地标记。
- 首次追加本节记录的补丁分段格式错误，未修改文件；修正 hunk 后成功写入。
- STEP-306 两个 Python 文件通过 AST 解析，Gitleaks 0 命中；本机未安装 Bash，Shell 语法检查跳过。生成独立提交 `5e162d5`。
- `codex/publish-local-project` 已成功推送并跟踪 `origin/codex/publish-local-project`；GitHub 已创建目标为 `main` 的草稿 PR。
- 收尾记录提交 `72b07c5` 已推送，本地与远端 SHA 一致；随后的状态核验发现并行本地流程新生成 3 个 STEP-307 源码脚本，继续纳入独立验证和发布。
- STEP-307 两个 Python 文件通过 AST 解析，Gitleaks 0 命中；提交 `fae5970` 已推送且远端 SHA 一致。最终状态仅剩明确排除的凭据、缓存、GBrain 本地标记与兄弟目录，没有未发布的 `DongFeng` 源码。

## Session: 2026-08-19 直接同步 main

- 用户明确要求本地需要保留的内容全部 commit 后直接 push 到 `main`；继续沿用已确认的产物排除范围。
- 只读召回分类为“直接适用”：直接更新 main、推送前复核范围和敏感信息、推送后核对 HEAD。
- 首次 `git merge --ff-only` 因本地 `remote_exec.py` 修改和未跟踪 STEP-307 文件而安全停止，未覆盖任何本地内容。
- 哈希审计确认发布分支之外真正新增/更新的源码为 `remote_exec.py`、`step307_poll.py` 和 3 个 STEP-308 脚本；其余未跟踪项均为排除产物。
- 上述 5 个文件通过 Gitleaks 和 4 个 Python AST 解析，形成独立提交 `b556349`。
- IPv4 扫描首次因 `rg` look-around 未启用 PCRE2 而无效；修正后只发现回环地址，非回环地址为 0。
- `git merge --no-ff --no-commit -X ours codex/publish-local-project` 自动完成且无未解决冲突；当前本地更新版本已保留。
- 合并树 152 个 Python 文件 AST 解析通过。全量 IPv4 检查发现两份历史 Markdown 各含 1 个非回环地址，未输出原值，已机械替换为 `[REDACTED_IP]` 并重新暂存。

## Session: 2026-08-19 全部原样提交

- 用户明确确认全部原样提交，不再匿名或排除缓存、二进制、训练/profiling 产物及明文配置。
- 原样范围限定为当前目标 Git 仓库；回复与 GBrain 仍不回传明文内容。
- 超过 GitHub 普通文件限制的对象计划使用 Git LFS，保持检出内容原样。
- 已删除此前新增的 `.gitignore`；下一步从本地 Git 对象库结构化恢复此前替换的历史值。
- 已从不可达 Git blob 中识别唯一历史值并恢复 3 个 WandB 配置文件及 2 个含历史地址的记录文件；恢复过程中未向终端输出值本身。
- 仓库盘点为 1745 个文件、约 1.30 GB；确认无嵌套 Git 仓库。超过 90 MB 的文件共 2 个（约 745 MB MSIX、约 474 MB profiling 压缩包），需要 Git LFS。
- Git LFS 3.7.1 已在仓库级启用，两个超限文件均生成 134 字节 LFS 指针；强制暂存后共有 1746 个跟踪路径、0 个未跟踪路径、0 个未暂存路径，暂存树中不存在达到 100 MiB 的普通 Git blob。
- 完整原样归档提交 `c012089` 已推送到 `origin/main`；两个 LFS 对象合计约 1.2 GB 均上传完成。远端与本地 SHA 完全一致，`git lfs push --dry-run origin main` 无待上传对象。

## Session: 2026-08-20 STEP-332 install query 实测

- 后 8 卡 30 步实测完成（`SOAP_STALE_Q_K=4`，带 `SOAP_INSTALL_DIAG` 埋点）。
- rank0 在 step 4/14/24 三次 install：共 1629 个因子，`query_true=100%`，`query_false=0`，`sync_ms=0`。
- iter4/14/24 仍 ~143s，per-factor event 优化无效。
- 结论：先前「install 时 QR 未完成需 synchronize」为错误推断，已被实测推翻。

## Session: 2026-08-20 Iter4/14/24 长尾定位与 soap.py 优化

- STEP-330：后 8 卡 `SOAP_STALE_Q_K=0 vs 4` 30 步 A/B 完成。k=0 iter14/24 仍 ~163s；证伪「仅 k=4 install 导致」假设。
- STEP-331：后 8 卡 rank0 profile（wait8/warmup1/active7），iter10–16 kernel_details 原位分析完成。**根因已用 trace 钉死**：iter14 内 `aclnnLinalgQr_QrAiCPU_Qr` 379 次串行，共 159.6s，占 kernel 总时 98.8%；iter10 同 profiler 窗口内 QR=0 ms。
- 原始 profile 文件已按规则删除（0 个文件验证通过）。
- 当前任务：针对 iter14/24 install 步同步阻塞，直接在 `soap.py` 设计优化方案。

## Session: 2026-08-20 推送审计后新增内容

- 只读审计确认既有 `main` 与远端一致、LFS 无待上传对象，但推送后新增 21 个未跟踪工具/缓存文件（合计 77,774 字节）及 `操作步骤.md` 的 47 行实质改动。
- `run_mapqr_876_train_inside.sh` 的规范化内容哈希与索引一致，仅为工作树状态/换行提示，不构成实质差异。
- 用户明确要求“全部推送”；本次保持当前 `main`，不创建或切换分支，并按明确路径逐项暂存。
- 暂存前发现并行本地流程已在 `main` 新增 1 个尚未推送的 NPU 性能提交，并改变了待提交文件集合；未覆盖或改写该提交，改为按最新状态重新取快照。当前已暂存 14 个实际路径，另有训练启动脚本在暂存后继续变化，需再次纳入。
- 最新暂存快照共 14 个路径，剩余实质未暂存和未跟踪路径均为 0；3 个新增 Python 源文件全部通过 AST 语法解析。训练启动脚本的规范化内容仍与索引一致，不产生可提交差异。
- 归档提交已连同此前并行生成的 1 个 NPU 性能提交一起推送至 `origin/main`。推送后本地与远端 SHA 一致，实质修改、未跟踪、被忽略的未跟踪文件及 LFS 待上传对象均为 0；训练启动脚本仍仅有换行/状态提示，规范化内容与索引一致。
## Session: 2026-08-21 STEP-338 最新提交与 loss 首分叉审计

- 只读核验远端 HEAD=`3a1d763`：提交本身只执行 MX QR → torch QR 回退。
- 发现远端 `soap.py` 有未提交 stale-Q/per-factor Event overlay；后续必须区分纯 commit 与 overlay，不能混合归因。
- 复用已有正式 30-step 日志，无新增训练：STEP-359 首个 loss 2% 越界点为 step 12（step11 `-1.3405%`，step12 `-2.3145%`）。
- torch/MX/两个调度实验均在 step12 首败，当前证据反对“step12 偏离由 MX QR 单独造成”。正在原位拆解 step10～12 子 loss 与 QR cadence，并准备从头构建无覆盖候选 kernel。
- 子 loss 已闭合：step12 差异约 90.92% 来自 map point regression 与 normal；torch 基线同型。
- 已纠正 QR 时序：iter11 末尾 Q 最早影响 iter13 forward，不能解释 iter12 forward 首败。
- STEP338 builder 已消除多层 shell quoting、OPP import 与动态 descriptor 三类夹具问题，并验证 `[192,192]` descriptor 可通过 para_check；真正进入 compile_op 后发现缺少 910B OpContext/SOC 初始化。当前仍无可用 object/json、未替换安装包。
- 已用官方 `opc`、正确 SOC/OpContext、custom tiling registry 和固定192 descriptor完成 retry4 编译；候选 object/json 唯一非空，安装包前后不变。
- 新逐step loss gate已通过8项测试，并在远端原位复核STEP359为11/30通过、step12首败；当前正审计候选隔离加载与命中证明，尚未执行设备A/B。
- 用户进一步明确：所有训练仅允许使用后8卡（visible 8～15）；已写入正式验收门禁。STEP343仍为单算子冷A/B，尚未执行训练。

## STEP-351：完整 shadow 冷测尚未形成上下文证据（2026-08-21）

- 已归档四类准备期错误：宿主/容器路径空间混用、本地AWK格式化错误、旧CLI参数名、旧STEP338 candidate stem硬编码。四者都发生在NPU/QR前，不计作算子失败。
- retry3首次真正越过shadow准备与后8卡8-rank live gate，但8 rank在release后失败；无done、无capture。controller/postflight与installed六文件复核说明进程已清场且客户安装件未被覆盖。
- 当前profiler证据不足以区分诊断AIC与原始AIC，实际traceback尚未完成原位读取。因此状态为`capture_invalid`，不是`operator_nan_reproduced`。
- 已暂停训练和重复冷跑。下一步只做一次只读failure定界；随后把同次A/Q/raw-R保存前移到decode前，并保留concrete AIC、header、四份T/V三个硬门禁。

## STEP-352：MX QrV2 正式修复实施方案进入审核（2026-08-21）

- 新增 `MX_QrV2_完整修复实施方案.md`，记录原始ZIP/installed只读保护、LocalTensor释放移动、zero-work tiling初始化、CANN const兼容、双SoC封装、CPU FP32/FP64 oracle、后8卡身份、30-step loss和相对原MX≤10%性能门禁。
- 固定正式策略：所有 `min(m,n)>80` 调用必须执行修复MX QrV2，不允许torch/CPU fallback；STEP350/351上下文抓取降为修复失败后的辅助诊断。
- 三项首轮独立只读审核已完成：源码修复方向无高危阻断；验收文档发现 `A_pad` 原地写回、矩形shape、oracle、身份门禁和性能基线未完全定义等问题。已逐项修订，正在做定向复审，仍未实施源码或远端变更。
- 三项定向复审全部PASS。文档已闭合源码后置断言、双SoC构建、公开API CPU oracle、original/fixed状态化A/B、concrete AIC身份、30-step loss和性能统计。状态为“方案可实施”，不是“修复已验证”；原始ZIP/installed/算子源码仍未修改。性能回归上限暂定建议10%，实施前待用户确认。

## STEP-357：正式修复候选已编译并封装为未验证包（2026-08-21）

- 本地补丁/归档工具17项测试、CPU oracle 20项测试通过；原ZIP与原cpp SHA保持不变。
- 目标容器CANN 8.3真实OPC已用`Ascend910_9362`成功编译candidate source `5a4d140b…4105b`，object 136856B、SHA `10ab542b…44cea`，JSON SHA `7ab6a53c…e97e9`。
- CANN 8.3 JSON实测无`kernelList`；验收改为顶层名称+object中NUL边界AIC/AIV精确集合，前缀/后缀/额外entry负例均拒绝。
- `Ascend910_93`与`ASCEND910B`同属DAV_2201，原wheel两目录产物逐字相同。因`--soc_version=Ascend910B`在当前工具链产生12个API错误却rc0，正式构建只编译真实`Ascend910_9362`一次，再以受审计alias-copy复制到`ascend910b`，两目录o/json SHA强制相同。
- installed 8文件、容器、OPC和CANN前后库存闭合；新包状态为`packaged_unvalidated`，尚未加载设备、未训练、不可发布。
- 下一步仅做后8卡8-rank核心状态化数学测试和raw profiler concrete AIC身份；通过后才进入30-step loss。

## STEP-362：动态 simplifiedKey 候选已重构并封包（2026-08-21）

- 已修正 STEP361 descriptor 中会使 OPC 取消 simplified key 的 `ori_shape/ori_format`；新 descriptor 仅保留 `shape=[-2]`、`format=ND`、`dtype=float32`，并保留 `--simplified_key_mode=0`。
- 本地 builder 21/21、CPU oracle 20/20 和独立审核通过。
- 唯一 STEP362 远端构建/packaging rc=0：`built`、`packaged_unvalidated`，installed/runtime inventory 均闭合。
- 新 wheel SHA256=`479bc12e5468d9ba60abc00d0a266b85265918fae465cbf77fb9aa6b4e018dd3`；未安装、未加载设备、未训练。
- 当前状态仍为未验证：待原位确认 dynamic metadata/concrete entry，然后进入后8卡8-rank 最小 identity+state 门禁。

## STEP-364：修复候选真实命中但首次192×192仍非有限（2026-08-21）

- retry5/6均在后8卡、8-rank、完整shadow中执行；8/8 raw task_track均实际引用candidate `QrV2_lifetime_fix_v1_0_mix_aic` 1次，original=0。
- retry5暴露失败摘要不兼容oracle早退schema；工具错误已经独立Recall、最小修复、4/4静态回归和独立审核。
- retry6真实谓词：首个STEP260 `192×192 FP32` 输入未改、输出shape正确，但finite FAIL；尚未进入状态序列。
- 后8卡、端口和受管进程清理PASS，installed 8文件前后不变。
- 结论：当前FreeTensor/zero-init/const组合补丁没有修复真实输入的NaN。已停止全量算子门禁和训练，转入剩余生命周期、最后panel与未初始化路径的二次源码审计。

## STEP-367：QrV2 v3 单核 MTE3→MTE2 最小候选（2026-08-21）

- 当前 v3 只在 `CalcQForLARFB()` 的 `vLocal` UB→GM workspace `DataCopy` 后、该 GM 作为 Matmul A 前新增一组 `MTE3_MTE2` Fetch/Set/Wait；未加入跨核 `SyncAll()`，未修改 `UpdateAForLARFB()` 或 base slot。
- candidate 唯一身份已升级为 `QrV2_lifetime_alpha_mte_fix_v3`，动态 concrete entry 为 `_0_mix_aic/aiv`。candidate SHA256=`fbfda044…13b99`；精确移除新增事件组后回到 v2 SHA256=`c4eef5c1…99c1`。
- CPU oracle 只增强 finite 早退归因字段与严格 JSON 失败摘要持久化，有限性总判据和全部数学阈值未变。
- 本地首轮发布/patcher/shadow 29项、CPU oracle 21项、STEP358静态7项全部通过。一次 `unittest` 路径参数错误发生在测试执行前，独立只读 Recall 确认为命令参数问题；改为直接执行测试文件后通过。尚未访问远端、未编译 OPC、未运行 NPU 或训练。
- 首轮独立审核确认算子增量通过，但发现有限 Q/R 的后续 FP32 诊断可能产生 `NaN/Inf` 并阻止严格 JSON，以及失败目录/traceback I/O 仍可能覆盖原异常。现已递归规范化非有限诊断标量，统一为不抛出的最佳努力持久化，并新增算术溢出及 mkdir/summary/traceback 三类故障注入。
- v3 未构建前，STEP358 控制器保持 `V3_RELEASE_READY=False` 和未设置 wheel/目录占位，在读取映射/连接前 fail-closed；外层 ZIP、worker docstring 和方案结论均升级 v3。最终本地测试 29+21+9=59/59 PASS；独立复核 P0/P1/P2 均为0。阶段性缺口仅为尚未执行 OPC/NPU/训练/性能外部验收。

## STEP-369：QrV2 v4 同步/所有权闭合候选（2026-08-21）

- v3 concrete AIC 已在 8/8 rank 实际命中但有限 192×192 输入仍产生非有限 Q/R，因此停止扩大测试并进入源码路径复审。
- v4 只叠加三项：`CalcQForLARFB` 使用 `m*blockSize + coreId*blockElement` 每核 scratch；`UpdateAForLARFB` 的 workspace DataCopy 与 Matmul B 之间补独立 MTE3→MTE2；core0 完成 Q 写回及 MTE3_V wait 后所有核无条件 `SyncAll()`，再各自释放 T/V/A。
- active LARFB 满足 `coreId < useCoreNum <= colNum <= blockp`，Process 额外调用仅 core0；scratch 末端不超过 `m*blockSize + blockp*blockElement = 2*m*blockSize`，未扩大 workspace。
- candidate 唯一身份为 `QrV2_lifetime_alpha_sync_fix_v4`，动态 concrete entry 为 `_0_mix_aic/aiv`。candidate SHA256=`2213dbae…614b`；三项精确反向后为 v3 SHA256=`fbfda044…13b99`。
- 本地发布/patcher/shadow 33/33、CPU oracle 21/21、STEP358 静态 9/9，总计 63/63 PASS；patcher check、py_compile 和限定 diff-check PASS。未访问远端、OPC/NPU/训练，原始 source 与 ZIP SHA 保持不变。
- v4 仍是未经过设备验证的候选。独立审核 P0=0、P1=0、P2=1；P2 仅建议后续增加对 offset 乘项和 DataCopy 长度的显式定向负例，固定 candidate SHA 与反向 v3 SHA 已能拒绝这些篡改，不阻断构建。STEP369A 当时控制器特意保持 `V4_RELEASE_READY=False`，用于防止旧 v3 wheel 冒充 v4；STEP370 构建封包后的当前状态见下节。

## STEP-370/371：v4 实施方案锁定与审核（2026-08-21）

- STEP370 已在目标 CANN 8.3 隔离目录完成 OPC 编译和新 wheel 封包，wheel SHA256=`4c158915bd5ae3fad4834a4f88028702d2d6fb534d69da45cd06f0b536f8dead`；未安装、未训练，installed/runtime inventory 前后闭合。
- 实施文档已锁定 8 级准入顺序：原件/范围 → 静态/构建 → 后8卡首个真实192输入 → CPU官方QR语义 → 状态化序列 → 30-step逐步loss≤2% → 耗时 → 交付/回滚。上一级失败立即停止，不扩大测试。
- STEP371 控制器已精确锁定 STEP370 wheel 路径/SHA和 v4 concrete AIC 身份，仅准备执行每 rank 一次 STEP260 真实 `192×192 FP32` identity+finite 核心门禁。
- 审核边界：目前只能表述为 `packaged_unvalidated`；concrete v4 AIC、Q/R finite、CPU语义、loss 与耗时都尚未通过。性能最大回归的精确百分比尚待用户锁定，10% 只是事前建议值。
- STEP370 发布物独立原位审计 PASS：wheel/外层ZIP/RECORD、双 SoC JSON/O、dynamic simplified key、v4 concrete AIC/AIV、binary-info 精确 QrV2 delta 和 installed/runtime 库存均闭合；device/loss/performance 仍为 pending。
- 代码审核 P0=0，C++ v4 修复未发现 P1；但 STEP371 host launcher 在 ownership 写入/身份获取异常路径上未统一进入 cleanup/postflight，记为设备测试前必须关闭的 P1。未修复前禁止启动 STEP371。
- 审核另发现方案与 `.codex-tools/step340_loss_gate.py` 的 zero denominator 语义不一致：方案规定 0/0 PASS，当前实现却无条件拒绝 GPU=0。该问题不阻断首个 finite 门禁，但必须在 30-step 前修正并补测试。
- STEP371 清场 P1 已关闭：`Popen` 后 identity/`getpgid`/ownership 写入/worker wait 的异常均保留首个错误，并无条件依次尝试 terminate、log close 和 cleanup/postflight；证据写入失败不再覆盖主错误。identity/getpgid/ownership 三类失败及 primary+terminate+cleanup 组合失败测试通过，STEP358 静态测试 11/11 PASS；独立复审 P0/P1=0。
- loss 零基线 P1 已关闭：GPU=0/NPU=0 定义偏差0并 PASS，GPU=0/NPU≠0 以 `gpu_zero_npu_nonzero` FAIL；missing/duplicate/nonfinite/普通相对偏差语义不变。定向测试 9/9 PASS，Python 审查 P0/P1/P2=0。
- STEP371 唯一真机门禁已执行并在 finite 层失败：8/8 raw profiler 均精确引用 `QrV2_lifetime_alpha_sync_fix_v4_0_mix_aic` 一次，其他 QrV2 引用为0；rank0 A finite 且 unmodified、shape PASS，Q nonfinite=36864（全矩阵），R nonfinite=16448，与v3失败计数相同。只有 finite 谓词 FAIL，其他数学谓词因早退未评估。
- STEP371 清场证据 PASS：preflight/LIVE_BINDING、物理设备8–15、postflight/finally 均闭合；端口34358 listener=0，后8卡进程=0，tracked launcher+8rank 共9个身份存活=0。rank1–7因全局终止无done/failure标量，不推断其Q/R。
- 裁决：v4 已确认真实执行但没有修复finite/NaN问题，立即禁止扩大到shape集、状态序列、训练loss或性能。当前转回源码路径审计，在形成可静态证伪的新单变量候选前不重跑设备。

## STEP-372：v5 Matmul tensor-position 方案记录与审核（2026-08-21）

- v4 的 R nonfinite=16448，恰等于 192×192 上三角总元素 18528 减首个64×64 R00上三角2080；这将首污染路径收敛到首次 LARFB。
- 两路独立源码审计确认：`vtvMatmulObj` 在 CalcQ 第二乘把 GM 交给声明为 VECIN 的 A；`qaMatmulObj` 声明 `VECIN/VECIN/GM`，实参却为 `VECIN/GM/VECIN`。这是已确认的静态合同错误，但是否解释全部真机 NaN 仍待设备证明。
- 新增 v5 fail-closed patcher 和测试，只有两个受控替换 hunk：CalcQ 第二乘直接 `SetTensorA(vLocal)` 并移除该次乘法专用 GM scratch/copy/event；`qaMatmulObj` 改为 `VECIN/GM/VECIN`。
- v5 SHA=`e6ccbb84b0e0dbdc026ecdc6b6e07936fbd659401e35c38f7e9eb974d99bc3b7`；反向两项精确恢复 v4 SHA=`2213dbae5027c179f09c8e3b43be51587ba22e3eb2a7546311048efc3844614b`。7/7 正反例、`py_compile`、patcher `--check`、新 Python diff-check 和 STEP372 文档区段行尾检查通过；Python 复审 P0/P1/P2=0。
- 实施文档已更新为 v5 唯一 identity、原件保护、两项 delta、OPC/封包/真192/CPU语义/状态序列/30-step loss/耗时/回滚八级门禁。
- 当前只准入 builder/shadow/worker/controller 的 v5 identity 接线和隔离 OPC 构建。尚未访问远端，未编译 v5，未运行 NPU/训练；不得宣称根因已完整证明或修复已完成。

## STEP-373A：v5 本地生成/构建/真机接线（2026-08-21）

- builder 已改为从 STEP372 v5 patcher 生成 candidate，`BIN_NAME=QrV2_matmul_position_fix_v5`，外层 ZIP 后缀唯一为 `-qrv2-matmul-position-fix-v5.zip`。manifest/_guard_tools 同时锁定 builder、v5 patcher 和其 v4 base patcher 依赖。
- shadow 除 v5 JSON/O/config/binary-info/concrete identity 外，新增 wheel 内 `qr_v2.cpp` SHA=`e6ccbb84…bc3b7` 硬门禁；worker profiler 仅接受 v5 `_0_mix_aic/aiv`。
- 新增 STEP373 隔离构建/封包 controller；远端上传清单精确包含原 ZIP、builder、v5 patcher、v4 base patcher，缺项/额外项/重名/SHA 漂移均 fail-closed。当前 `BUILD_READY=False`，独立审核前不能连接或构建。
- STEP374 设备 controller 已先解除 v4 武装：`V5_RELEASE_READY=False`、wheel path/SHA 为 `None`，且显式拒绝已知 STEP370/v4 路径和 wheel SHA，ready 失败发生在读取机器映射之前。最终 summary 显式要求 8/8 rank 的 input-unmodified、shape、finite、reconstruction、orthogonality 及 concrete AIC 全部通过。
- 本地验证：v5 patcher 7/7、release tools 36/36、STEP358/374 静态 13/13、STEP373 wiring 5/5 均 PASS，相关 Python `py_compile` 和 patcher `--check` PASS。原 source/ZIP/v4 patcher SHA 保持不变。
- 当前边界：本地构建链审核 P0=0/P1=0 后已将 STEP373 `BUILD_READY` 改为 true，dry-run 通过；唯一远端构建结果见下节。NPU/训练尚未执行。

## STEP-373B：v5 隔离 OPC 构建、封包与独立原位审计（2026-08-21）

- 按远端安全规则重读两份映射，末段42、两跳、二跳 hostname 和精确运行容器 `mapqr-leicheng` 通过。唯一 STEP373 构建 rc=0，未安装、未调用 NPU、未训练。
- 新 wheel SHA=`f20c3db839b669ef6919b2a40df80b475676a9c0149910e0c73eb65064b8c11b`、size=1,732,412；outer SHA=`be863ff1f947edd21650fa093740bf6ed6363cfb10bd2e8521219dee96be8567`。candidate source=`e6ccbb84…bc3b7`，kernel=`QrV2_matmul_position_fix_v5`。
- 独立 Agent 一次只读原位复算 PASS：wheel/RECORD 247/247；双 SoC JSON=`7468d070…a4ad3`、O=`fa74fade…c4154`/141008 bytes，字节一致；dynamic `[-2]`、`simplifiedKeyMode=0`、2 keys、唯一 v5 AIC/AIV、config/binary-info、manifest 三工具 SHA、installed/runtime inventory 全部闭合；v4 candidate artifact 引用为0。P0=0/P1=0。
- 真机 controller 的两个审核 P1 已关闭：SUMMARY 现在显式锁定每 rank 固定192输入 SHA/FP32/MX分支/内部 `_C.qr` 输入/后8卡/raw profile/数学谓词；外层连接/超时异常只使用 ownership PID/starttime/PGID 协议有界清理，原始错误优先。18/18 执行型和负例测试通过。
- 发布物审计通过后，STEP374 已精确接入新 wheel 路径/SHA、新端口34359 并设 `V5_RELEASE_READY=True`；dry-run 通过。当前等待 phase-transition 最后独立复核；设备门禁仍未执行。

## STEP-374A：v5 单次真机核心门禁启动失败（2026-08-21）

- phase-transition 独立复核 P0=0、P1=0，只准入一次 STEP374；运行前重新核对两份主机映射，目标末段42、二跳身份、精确容器及 wheel SHA 均由控制器继续硬校验。
- 唯一一次执行返回 `host gate rc=122`，没有结构化 SUMMARY；未复跑，也未进入 CPU 语义、训练 loss 或性能阶段。
- 源码复核确认 `122` 是 host controller 的统一 fail-fast 码，不等同于超时：ready/done、rank failure、ownership、进程等待、终止或日志关闭异常均可能映射为该码。当前不能据此裁决 v5 数值是否通过。
- 已启动独立只读远端证据审计；原始 profile 和诊断目录保留，不下载、不删除、不运行新的 NPU 工作负载。故障 Recall 分类为 world8 控制器 fail-fast 汇总码，receipt=`unavailable_no_worker_token`。

## STEP-374B：v5 runtime-invalid 只读证据闭环（2026-08-21）

- 独立 Agent 重新核验两份映射、目标末段42、二跳 hostname 和精确容器后，只读审计远端原始目录；未写、未下载、未运行NPU或训练。
- controller=122、launcher=1、postflight=0；LIVE_BINDING_PASS，ready=8/8、done=0/8、failure traceback=8/8、failure scalar=0/8、profiler identity=0/8。
- 8/8 均在第一次真实192调用的同步阶段报507014/AICore timeout/trap，故障kernel均为v5 concrete AIC；日志含MTE错误，部分rank另有AIVector exception。
- v5 命中已由设备错误身份确认，但 profiler identity 未生成；Q/R 未回传，不能判断数值。当前候选裁决为 `runtime_invalid`，不得进入CPU语义、训练loss或性能。
- 清场闭合：postflight/finally PASS，端口34359监听0、后8卡进程0、9个受管PID/starttime存活0；原始证据保留。
- 文档独立审核初次结果 P0=0、P1=3、P2=3：旧当前时态、遗漏qa位置变更、后续门禁误接v5三个P1均已修正；同时区分“设备错误点名内核进入执行”与“profiler identity PASS”，并将probe收紧为单观察边界/单可逆delta、独立identity/SHA及显式workspace/tiling/event增量。
- 修订后独立复审 P0=0、P1=0、P2=1；残余P2仅为STEP372历史区段未显式失效。已将相关标题、推测、未验证项和旧T/V/q/a分支标为历史，并回链STEP374/第15节当前裁决。

## STEP-375A：离线PC映射与单变量生成器（2026-08-22）

- 远端只读PC审计重新核验两份映射、末段42、二跳hostname和精确容器；未写、下载、运行NPU或反汇编生成文件。ELF函数边界与运行时pc_start精确闭合，但无DWARF/line map，只能得到AIC两簇offset和AIV统一offset，不能映射源码行。
- 两路本地源码审计确认delta1没有静态越界/use-after-free/跨核地址错误；delta2会改变qa的GM读和Local写lowering，现有同步证据不足，是P1证据缺口而非已确认源码错误。
- 新增STEP375 delta1-only诊断patcher/test，identity=`QrV2_vtv_direct_qa_legacy_probe_v6`，candidate SHA=`ef5db14e…ce180`；只保留vtv direct-vLocal，qa明确保持legacy `VECIN/VECIN/GM`，`diagnostic_only=true`、`release_candidate=false`。
- 主流程独立重跑8/8测试、patcher `--check`、AST和原件SHA均PASS；原`qr_v2.cpp`、v4/v5 patcher未变。当前等待独立代码复审，未接builder、未远端构建、未运行NPU。
- STEP375独立代码审核首轮P0/P1=0、P2=2；补齐symlink/CLI/成功写/写失败清理测试并保留首异常后，复审又发现修改`exception.args`会改变`str()`的P2。最终改为只附加`cleanup_error`、能力探测`add_note`且失败不覆盖首因，测试增至18/18。
- 最终主流程18/18、`--check`、AST通过；独立复审P0/P1/P2=0。patcher SHA=`98a655f8…11768`，test SHA=`6c064afa…72cd5`。允许进入独立diagnostic builder设计，仍不允许远端构建或NPU。
- STEP376设计审核确认直接复用STEP373/v5 release builder是P0风险；当前开始实现仅支持prepare/build、禁止package/all/wheel的diagnostic adapter。
- STEP376 adapter/test已新增。首次测试因猜测v4依赖文件名失败，未触发远端或修改既有文件；按故障Recall改为从STEP375的`release_v4.__file__`解析并锁目录/SHA，receipt=`unavailable_no_worker_token`。
- 首版11/11后独立审核发现base module仍可调用package/all的P1、seal前后artifact闭包P1及approved-root责任未编码P1。已毒化base release API，强制唯一`approved_root/work`，并在seal前/落盘后复算object/json/opc_log、identity、双SoC SHA和bytes；测试增至15/15。
- 最终adapter SHA=`fc65fecc…c299e`、test SHA=`6eef234d…bfde1`；主流程17/17和AST PASS。alias分叉与seal前窗口篡改定向负例已补齐，独立复审P0/P1/P2=0。
- STEP376专用远端构建controller骨架已实现：只上传精确6个普通文件，只允许diagnostic adapter的prepare/build；独立采集构建前后installed/runtime/相关进程快照，容器命令受GNU timeout约束，summary采用严格schema和路径/SHA闭包，异常路径保留首错并依次尝试清理。失败证据为有版本、字段/数量/长度受限的单行JSON，恶意异常对象也不能覆盖主错；全部Python入口禁止写bytecode。controller SHA=`dac8e26a…ee20e`、test SHA=`a72edf4f…b9a74`；主流程42/42、`py_compile`、AST PASS，独立终审P0/P1/P2=0。
- phase-transition后唯一一次远端执行在exclusive目录检查时报`[: missing ']'`并停止，未自动重试。只读现场审计确认目标诊断目录不存在、OPC/adapter相关进程为0，installed/runtime未被本次触达；精确根因是旧脚本在quoted path与`]`间缺空格。
- controller已回退`BUILD_READY=False`并以`_exclusive_directory_script()`修复；新增bash语法、特殊路径、重复/悬空symlink、20轮8路竞争和execute级无上传/无重试/清理保真测试。当前controller SHA=`572a652a…c5e48`、test SHA=`672219f9…5981`，46/46 PASS。必须重新phase-transition才能有新的远端构建尝试。
- retry2 phase-transition允许以全新目录重新武装。当前DIAG_NAME=`step376_retry2_qrv2_vtv_direct_qa_legacy_probe_v6_build_20260822`、`BUILD_READY=True`；candidate/输入SHA/adapter/权限均未变化。controller SHA=`8160238c…c7e3`、test SHA=`c1000979…2fa0`，46/46 PASS；尚未执行attempt2，等待armed-state终审。
- retry2 armed-state终审P0/P1/P2=0后唯一执行失败并停止：首个before-snapshot把官方`latest/bin/opc` symlink alias传给base builder，后者按合同拒绝symlink，故work/manifest/OPC产物均为0且OPC未启动。当前转入controller realpath合同修复，禁止重试。
- realpath最小修复已完成：controller只接受合同中非空、绝对、无NUL的`opc.path`，并统一传给pre/post snapshot和container build；legacy OPC alias不再进入命令，base/adapter non-symlink regular/path/SHA门禁保持不变。当前`BUILD_READY=False`，controller SHA=`e7d397ef…d3598`、test SHA=`ef2b9cd2…d7147`；49/49加OPC SHA定向1/1 PASS，独立审核P0/P1/P2=0。
- attempt3 phase-transition后已用全新DIAG_NAME=`step376_attempt3_qrv2_vtv_direct_qa_legacy_probe_v6_build_20260822`武装，`BUILD_READY=True`；测试反拒attempt1/retry2名称，candidate/输入SHA/adapter/权限未变。controller SHA=`426b9a45…0a1d3`、test SHA=`f10793a4…3ae4`，49/49加定向1/1 PASS；尚未执行远端，等待armed-state终审。
- attempt3 armed-state终审P0/P1/P2=0后唯一构建成功，返回`diagnostic_built_unvalidated`。双SoC object SHA均=`a75ff58a…be14`、JSON SHA均=`c34a02cb…c490`且bytes一致；concrete AIC/AIV identity正确，package状态为forbidden，release输出不存在，installed/runtime闭包PASS。
- 构建后已立即回退`BUILD_READY=False`，controller SHA=`1bd72bc9…2b7a5`、test SHA=`70b453ab…b342`，49/49 PASS。随后使用同一锁定helper环境独立原位重算summary与snapshot：hostname/container PASS、相关构建进程0、全部SHA/flags/闭包再次PASS。未下载、未运行NPU/训练。
- STEP377只读设计完成：以immutable原wheel安全解包完整shadow，仅在shadow内替换双SoC QrV2 object/JSON并更新两处必要路由；不复制source、不改RECORD、不生成wheel/zip/release、不安装。worker只允许每rank一次固定STEP260真实192×192 profiled调用；专用controller默认`NPU_READY=False`，后8卡/identity/task-reference/finite/math/清场均为硬门禁。当前开始仅实现shadow builder及负例，尚未接NPU。
- STEP377 shadow builder及测试已完成三轮独立审核。首审P0=1/P1=3/P2=2、二审P1=2/P2=1、终审前P2=1均已关闭；最终P0/P1/P2=0。最终builder SHA=`bb080a82…356ff`、test SHA=`1c804276…0a938`，16/16 PASS。当前仅本地fixture验证，尚未接远端attempt3 manifest、worker或controller。
- STEP377 diagnostic worker薄适配器已实现并终审P0/P1/P2=0：锁底层worker/cold/oracle SHA，强制唯一`--first-profiled-only`，task-reference全集严格为diagnostic AIC=1且多hash同identity直接拒绝；diagnostic gate以new regular/O_NOFOLLOW/双lstat inode闭包替代release语义；所有patch/sys.modules/argv恢复主错优先。最终adapter SHA=`1be60dd3…96519`、test SHA=`16d1a891…ce53`，12/12 PASS。尚未接host/controller或NPU。
- STEP377 worker+host协议经两轮加固后终审P0/P1/P2=0、30/30 PASS：gate token由worker每rankO_EXCL ack并锁dev/inode/token SHA，host收齐8 ack并在done后复验；ready nested schema和跨rank模块SHA一致；失败路径输入SHA finally、数学done exact schema、后8卡/live binding及STEP358 ownership cleanup均闭合。最终worker SHA=`f363ac8b…e4c3d`、host SHA=`6b150f35…371a`。
- remote controller当前仍`NPU_READY=False`。v3已修正依赖闭包、attempt3 manifest路径、summary input SHA并接入真实STEP357连接原语，但容器shadow/snapshot/host/summary/forbidden/cleanup仍是action envelope placeholder；独立审核P0=0/P1=4/P2=2，禁止武装或远端执行。
## 2026-08-21：Phase 17 启动

- 用户明确最终目标为：同抓取时机/步骤的模型内 MX QR 与官方 CPU QR 对照，加上同一输入的 MX/CPU 单算子 A/B，并要求保留可提交算子部门的问题输入输出证据。
- 已只读核验 npu2（末段171）：目标身份 PASS，精确容器 `mapqr-leicheng` 运行中；NPU 0–3 有 Python 任务，NPU 4–7 无进程，当前不满足8卡正式测试门禁。
- 用户提醒另一个项目正在共享代码上修改/训练。已冻结隔离策略：不修改活跃共享工作树，不切其分支；后续只使用固定提交的唯一隔离副本/独立 worktree、独立端口和输出目录。
- 重大隔离决策只读 Recall 返回分类“共享工作树隔离与远程测试安全”，约束与上述策略一致；receipt unavailable。
- 当前继续推进本地夹具审计与实现，不因等待8卡而停止；尚未启动训练、算子或写入远端。
- 用户追加硬要求：必须在真实端到端训练调用中抓取算子输入输出，并及时落盘；离线单算子 replay 不能替代主证据。已同步到 Phase 17 验收项。
- 首次把 npu2 身份、容器挂载、活跃训练、仓库身份和 NPU 进程合并为一个只读 expect 调用时，工具返回 exit0 但 stdout/stderr 均为空；无远端成功标记，不能视为查询成功。独立 Recall 分类“远程执行可观测性异常”，后续拆分为带唯一成功标记的小命令，不重复原组合路径；receipt unavailable。
- 拆分复验成功：npu2 hostname/IP、精确容器与挂载通过；共享仓库 HEAD 为 `3a1d763...`、分支 `ascend_npu_optimize`、11 个 tracked dirty 路径，SOAP 已有其他项目未提交诊断改动。宿主查询时未发现 `train_spetr/torchrun/ddp_train` 命令，但尚未重新核验全部 NPU 进程，不据此判定8卡空闲。
- 用户明确固定使用后8逻辑设备（8–15）。已纠正先前把 NPU 4–7 四个双-chip组误称为“仅4卡空闲”的表述；上次 `npu-smi` 实际显示 Phy-ID 8–15 均无进程。正式启动前仍需即时复核。
- 定向读取共享 SOAP 段的第二个只读 expect 命令未取得 `QUERY_OK_SOAP` 标记，按门禁判无证据；未写远端。后续不依赖该活跃脏文件，改从固定 Git commit 的只读 archive 构建隔离副本。
- 已新增本地训练态捕获模块、严格 SOAP 补丁器和单测：输入在目标算子调用前先原子保存；MX/CPU 实际输出返回后立即保存；异常抛出时保留 input+failed record；CPU 路径明确对 CPU tensor 调官方 `torch.linalg.qr` 后回送原设备。首次 unittest 命令误带 `.py` 被解析为属性，改模块名后执行；静态夹具缩进与生产锚点不符导致 1/6 失败，修正夹具后最终 6/6 PASS，三文件 `py_compile` 与 `git diff --check` PASS。
- 写后代码复审发现正式证据仍缺 world8/后8设备硬门禁和 source/config/checkpoint 身份字段，正在补强；尚未部署或运行远端。

## STEP-377E：controller文件闭包实现与独立审核失败（2026-08-22）

- 子任务仅修改STEP377 controller及测试；主流程`py_compile`和19/19测试PASS。controller SHA=`3f375469c21ed42664eb982535bdadd1f4ac9e4857355de4a93f2917e0d0708b`，test SHA=`26adcdd576f87d6171ed5ebc2d9e34614d42d76e736ae1d36be69318b4a6ba84`。
- 独立只读审核裁决P0=2、P1=3：route路径穿越；shadow artifact/RECORD与原wheel树未精确绑定；upload目录项读取后闭包不足；本地上传源有hash/read替换窗口；installed QrV2扫描范围不完整。
- `NPU_READY=False`保持；未读取机器映射、未连接远端、未运行NPU/训练。下一步仅关闭上述P0/P1，再处理PID复用安全清场与稳定双采样；复审通过前禁止phase-transition。

## STEP-377F/G：文件闭包、安全清场与rank ownership全链收口（2026-08-22）

- 文件/route/shadow/upload闭包经过多轮独立复审，已强制实际wheel路径与整包SHA、attempt3双SoC artifact SHA、config精确route变换、唯一RECORD、全树类型与首尾目录身份、全customize QrV2语义库存；终审P0=0/P1=0。
- 新增STEP377 process guard：双次真实grammar `npu-smi`采样、同一`/proc/PID` dirfd身份、严格rank/device/NSpid双射、pidfd-only信号、launcher PGID授权及固定8-rank identity清场。launcher未确认但发现approved残留时0 signal并failclosed。
- rank ownership在gate发布前以O_EXCL临时文件、file fsync、hardlink commit及目录fsync持久化；gate同样以hardlink为commit point。postlink目录fsync失败明确属于“已发布但耐久性失败”，仍保留可读证据并进入四域cleanup，不误报未发布。
- rank、launcher、stable-clear、port四个清场域始终全部尝试再聚合错误；ownership、rank和gate均以expected SHA/case/port/token端到端绑定。旧STEP343裸PID/killpg cleanup在STEP377生产路径中不可达。
- 最终SHA：guard=`f63813d6c7f590bc4ce9a45a58901ee9192b11b86d2783e2cae05d135bbb0490`，host=`4be8541aafdb727c83273442fc2a14dc9079911b89ae2afcdb6797d4669854e4`，controller=`c125725ef11abc602c2d7fd518384fe8c7b06ef029200b2cb79822ee93b303d2`；对应测试SHA=`0a5bf805…2bc5`、`134abb6f…b445`、`74d89a09…0b37`。
- 主流程guard21/21、host21/21、controller27/27、worker14/14、shadow16/16均PASS，`py_compile`和diff-check通过。两路最终独立复审均为P0=0/P1=0，仅剩读取上限一致性、未使用旧函数和额外故障注入等P2维护项。
- 当前仍为`NPU_READY=False`，未运行STEP377 NPU诊断或训练。下一步是独立phase-transition审核，不得把本地绿测称为修复成功。

## STEP-392/Phase23：转入真实训练验证（2026-08-22）

- 完成standalone delta2-only后8卡world8；8/8 identity/finite/QR数学PASS，但按用户纠正不将其称为端到端修复成功。
- 已暂停release promotion和cleanup薄适配实现；中断的子agent未新增STEP393文件。
- 已盘点STEP204 formal 30-step、GPU loss oracle、`step340_loss_gate.py`、STEP392 shadow和历史capture工具。
- 当前正在锁定唯一MX QR基线commit和低扰动30-step实施合同；锁定前不运行训练。
- STEP393 本地夹具已进入 P0/P1 收口：新增文件保持默认 disarmed；唯一聚焦组修正陈旧断言后1/1 PASS。独立审核判当前 NO-GO，未远端、未训练、未重复 standalone。下一步只关闭 bootstrap pidfd/ownership、原子 handoff、生产环境保持与严格结果交叉校验，再复审。
- STEP393夹具最终P0/P1=0后执行attempt1；SSH/身份/排他目录/上传通过，但archive首个shell test因缺空格报`missing ]`，未进入训练/NPU。错误已独立分类并最小修复；旧目录不复用，attempt2待增量审核。
- attempt2在archive `_host` 前被本地全脚本分号预检误伤，仍未训练/NPU；已改为纯helper局部token自验并通过唯一聚焦组。按约定不自动attempt3。

## 2026-08-22：STEP393 attempt3 与进程守卫修正

- attempt3 在首个进程组观察中被无关 PID2 的 `pgrp<=1` 严格解析误停；远端审计证明8 rank未ready、训练未start、无loss/耗时结果，当前case/PGID/端口/后8卡均清零。
- 仅修改STEP393 stat预筛，非目标PGID先行忽略，目标PGID继续严格fail-closed；STEP377未改。
- 3项必要guard测试与`py_compile`通过，正在等待独立Python代码审查；审查通过前不运行attempt4。

## 2026-08-22：STEP393 attempt4 与摘要链修正

- attempt4在pre-snapshot因backend内嵌旧guard SHA确定性停止，未创建run目录、未启动rank/训练；现场连续双零清场。
- 仅同步backend guard锁和controller backend锁，新增三方SHA交叉断言；focused tests和独立Python审查P0/P1=0。
- 下一候选目录改为fresh attempt5，源码双门禁仍False，等待phase-transition复核。

## 2026-08-22：STEP393 attempt5 静态合同停止

- attempt5在8 rank ready前由runner静态config SHA门禁停止；无训练/MX QR/loss结果，cleanup postflight及双采样清场PASS。
- SOAP锁定实际PASS；审计初报SOAP失败为NUL转义假阳性，已纠正。
- 既有STEP193/204 config从锁定`02aca0...`漂移为`79c014...`，attempt5未生成或修改它。当前只读查找canonical副本，不放宽SHA、不创建attempt6。

## 2026-08-22：STEP393 attempt6与交接

- canonical config已在新文件中精确恢复并通过独立集成审核；attempt6启动后0/8 ready timeout。
- 原位审计定界到torch/torch_npu import链ImportError，未到Config.fromfile/launcher/rank/MX QR；现场清场PASS。
- 已生成`MX_QrV2_端到端问题交接文档.md`。下一步只修preflight失败可观测性与host waiter早停，不复用attempt6、不重跑30-step。
