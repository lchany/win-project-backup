# Progress Log

历史全文已备份到 `planning_backup/2026-08-18/`。本文件从 2026-08-18 规划文件轮换后重新开始。

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
