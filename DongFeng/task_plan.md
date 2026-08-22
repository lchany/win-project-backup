# Task Plan: Ascend NPU 性能优化

历史全文已备份到 `planning_backup/2026-08-18/`。本文件只保留当前有效目标、门禁和下一步。

## Goal
在不改变最终功能、训练语义且保持 loss/梯度门禁的前提下，为 `ascend_npu_optimize` 做可复现、可量化、按功能独立提交的昇腾 NPU 性能优化。验收：同合同下 8 卡 NPU 与 8 卡 GPU 的 `throughput (samples/s)` 达到 1:1 或更好。用户 2026-08-18 双门禁：耗时相对 CPU FP64 SOAP 基线大幅下降，且逐步 logged `loss` 相对 GPU `|Δ| ≤ 2%`。

## Next Step
先关闭 `MX_QrV2_完整修复实施方案.md` 第17节审核问题，并由用户在性能测试前锁定 fixed/original MX 允许的最大耗时回归。P0/P1归零后，才可审核一次 STEP377 delta1-only 诊断；它只能判断 v5 trap 的 delta2 必要性，不能直接选定 release delta。

## Current Phase
**Phase 16 repair-plan review in progress**：当前只审核正式修复实施方案；不修改算子、不连接远端、不启动 NPU 或训练。顶层新步骤不再指向历史 npu2/Phase17 路径。

## Standing Rules
- 每次远端访问先核验 `机器IP.md` 中主机器末段为 `42`；默认只访问主机器。仅当用户明确指定 npu2 时可访问登记的末段171目标，并在第二跳连接后核验 hostname 与目标 IP；不得访问其他机器。
- 训练/基线/A/B/profiler 只在完整名称 `mapqr-leicheng` 的现有容器中执行；8 张昇腾 NPU，禁止 GPU/CUDA/CPU 代替训练。
- 禁止改远端驱动、固件、CANN、PyTorch、torch_npu 和既有依赖版本。
- 禁止把远端数据、日志、profile、权重、checkpoint 拉到本地；分析必须远端原位完成。
- 提交前缀：`【npu性能优化】<对象与动作>`；同一功能原则上一个 commit。
- 永久算法基点：`63861df 【loss对齐】随机性移除`。累计收益必须相对该提交；相对 HEAD/父提交只作增量辅助。
- 正式修复禁止 `git restore` / overlay 抹掉 `fb979b2` 历史；若修精度须新 commit。
- 远程机器信息只从本地 `机器IP.md` 读取，不写入本文件。

## Phases
### Phase 1–7: 审计、随机性移除、基线、算子替换、系统优化、回归、新机器恢复
- **Status:** complete（历史细节见备份）

### Phase 8: SOAP 精度合同与社区 QR
- [x] 确认 NPU/GPU 分叉可归因于 `fb979b2` 的三条优化器语义变更（one-sided=1024、identity 初基、NPU FP32 周期栈）
- [x] 按 2% 门禁重评分既有 30 步对照：只有 63861df CPU FP64 双轴达 30/30
- [x] 拆开社区 QR NaN 与 11/30 精度两个独立根因（STEP-265）
- [x] 对照 CPU FP64 SOAP，判定 NaN 是 mx QR 最后 64 列 tile 输出缺陷，不是 SOAP I/O（STEP-266）
- [x] 8 个 BAD `.pt` 已落到本地并传到同事机 `/home/ubuntu/`
- [x] STEP-268：192×192 加严复现（原 A、幅值、邻域尺寸、8 卡 replay）
- [x] STEP-269：布局/warmup/8 卡隔离 + 对照 `qr_v2.cpp` 最后一块 tile
- [x] STEP-270：设备分域（npu2–7 全 shape>80 崩）+ npu0/2 对照矩阵
- [x] STEP-271/273：192×192 bypass 到 `torch.linalg.qr`，前后 8 卡 16/16
- [x] STEP-274：后 8 卡 bypass 完整 30 step，loss 30/30 ≤2%，无 NaN
- [ ] QrV2 `.o` 重编后关闭 bypass 复验；SOAP 周期加速（Iter4 208 s）
- **Status:** in_progress

### Phase 9: 本地项目安全发布到 GitHub
- [x] 核对仓库根目录、分支、远端及全部在途文件
- [x] 安装 GitHub CLI 并检查认证状态
- [x] 审计敏感信息、缓存、训练/profiling 产物及超限文件
- [x] 建立发布分支并提交首批安全范围（`0fa5e1b`）
- [x] 验证并提交首批提交后新出现的 STEP-306 源码脚本（`5e162d5`）
- [x] 推送分支并创建草稿 PR
- [x] 验证并提交 STEP-307 源码脚本，推送后复核（`fae5970`）
- **Status:** complete

### Phase 10: 按用户明确要求直接同步 main
- [x] 只读召回并确认继续排除二进制、缓存、训练/profiling、凭据和本地状态
- [x] 哈希审计当前本地源码相对发布分支的增量
- [x] 提交 `remote_exec.py`、更新后的 STEP-307 poll 与 STEP-308 三个脚本（`b556349`）
- [x] 无冲突合并 `codex/publish-local-project`，保留当前本地增量
- [x] 最终安全门禁、提交合并记录、直接推送 `main` 并核验 SHA
- **Status:** complete

### Phase 11: 全部原样提交
- [x] 用户明确确认覆盖此前脱敏与产物排除选择
- [x] 启用 Git LFS 方案处理超过 GitHub 普通文件限制的文件
- [x] 从本地 Git 对象库恢复此前被替换的历史 IP/WandB 原值
- [x] 删除 `.gitignore` 并盘点仓库当前全部文件
- [x] 配置 LFS、强制暂存并核验全部对象
- [x] 提交并直接推送 `main`，核验远端 SHA 与 LFS 上传状态
- **Status:** complete

### Phase 12: 推送审计后新增内容
- [x] 只读核对本地/远端 `main`、工作区、分支与 LFS 状态
- [x] 用户明确授权全部新增与改动直接推送
- [x] 按已确认路径暂存并验证提交范围
- [x] 提交、推送并重复核验直到工作区无实质差异
- **Status:** complete

### Phase 13: 读取当前项目目录内的相关本地文件
- [x] 恢复并读取现有规划文件索引
- [x] 盘点项目根目录 Markdown 文件与规模
- [x] 读取 `AGENTS.md`、脱敏解析 `机器IP.md`
- [x] 收到用户范围修正后停止读取当前项目目录外文件
- [x] 从大型 `操作步骤.md` 中提取 STEP-315～335 当前主线记录
- [x] 汇总已确认事实、合理推测与未验证假设
- **Status:** complete

### Phase 14: 验证 MX QR 问题根因
- [x] 重审 STEP-319～323 单算子与 MX/torch 对照方法
- [x] 重审 STEP-315 dump 边界、调用顺序与 SOAP 下游首次分叉
- [x] 核验远端最新 HEAD 与未提交 overlay，确认当前运行不能直接归因于最新 commit
- [x] 用现有正式日志定位总 loss 首个 2% 越界点为 step 12
- [ ] 建立 QR 合法性、子空间/规范化和 SOAP 状态连续性三层 oracle
- [ ] 设计最小远端实验，核验设备、stream、调用序列与 first divergence
- [ ] 在 8-NPU 正确容器中执行并形成根因裁决
- [ ] 修复候选必须通过逐 step GPU loss 相对偏差 ≤2% 硬门禁；任一步超限即拒绝
- [ ] 用户明确耗时比较对象与允许回归阈值后，才可对修复作最终 PASS 裁决
- **Status:** in_progress

### Phase 15: 训练态 QrV2 上下文抓取
- [x] 冻结证据合同：同一次调用关联 A/Q/raw-R 与 Free 前后 T/V；缺项即 INVALID
- [x] 编译保留原 `FreeTensor()` 顺序的 STEP350 诊断内核
- [x] 记录 STEP351 shadow 冷测的准备期错误和首次真实执行失败
- [ ] 原位读取 STEP351 retry3 的8-rank首个 traceback，确认失败发生在 QR、profile 还是 decode
- [ ] 冷测必须同时通过 concrete AIC、header、四份 T/V 完整性门禁
- [ ] 冷测通过后仅执行一次最小训练窗口，抓取真实 NaN 调用上下文
- [ ] 用同一调用证明或否定 `FreeTensor` 后 T/V 异常与 Q/R 非有限区域的因果链
- **Status:** in_progress_fail_closed

### Phase 16: MX QrV2 正式修复方案与审核
- [x] 将原件保护、源码修改、CPU oracle、设备身份、训练和性能门禁写入独立实施文档
- [x] 固定纯 MX 策略：所有 `min(m,n)>80` 调用禁止 torch/CPU fallback
- [x] 完成源码生命周期/同步独立审核
- [x] 完成CPU语义、矩形模式、性能与回滚首轮独立审核
- [x] 根据审核发现修订文档并形成最终审核结论
- [x] 实现fail-closed补丁器、双层ZIP/Wheel构建器和CPU oracle
- [x] 在目标CANN隔离编译DAV_2201候选并生成未验证新包，原件/installed不变
- [x] 实现完整shadow、8-rank concrete AIC、调用账本、80-case数学与owned-PID清场门禁
- [ ] 在后8卡8-rank验证candidate concrete AIC和状态化数学合同
- [ ] 通过30-step逐步loss与性能门禁后再标记可发布
- [x] 重新固定当前实现方案：历史 v1～v5/诊断产物不得直接升格，先用设备证据选择唯一最小可逆 release delta
- [x] 当前实施方案独立审核达到 P0=0、P1=0
- **Status:** repair_plan_review_passed_waiting_step377_phase_review

### Phase 17: 同条件端到端与同输入单算子 MX/CPU QR A/B
- [ ] 审计现有 STEP-319～323、release oracle、训练 QR 调用点与 dump 格式，形成逐项证据缺口表
- [ ] 实现同一 rank/step/call-index 的真实端到端训练双路径捕获：必须在 SOAP 实际算子调用点进入前抓 A、返回后抓 Q/R；未命中目标算子即 INVALID
- [ ] 实现同一现场输入 A 的 MX/CPU 单算子重放，包含 QR 符号对齐、重构、正交、三角和子空间 oracle
- [ ] 每次目标调用结束立即原子落盘 A/Q/R 与 manifest，记录 SHA256、shape、dtype、stride、finite/NaN/Inf 和运行环境指纹；异常调用不得等待训练结束才保存
- [ ] 在 npu2 `mapqr-leicheng` 内使用固定 commit 的唯一隔离副本；禁止修改活跃共享工作树，独立端口和输出目录
- [ ] 固定后8逻辑设备 `ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15` 执行 8-rank 测试，核验 `torch_npu`、rank/PID/Phy-ID 8–15 一一绑定及测试前后无残留
- [ ] 原位生成算子部门可独立重放的脚本、原始坏输入输出、CPU 对照与脱敏汇总；不得拉取远端产物到本地
- **Status:** in_progress

## Acceptance Criteria
- 逐步 logged `loss` 相对同一 GPU log 的相对偏差 `|NPU-GPU|/|GPU| ≤ 2%`；任一步超限即失败，不以均值替代。
- 所有训练、A/B 训练和 profiler 训练仅使用后 8 卡：`ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`，并核验 8 rank 与 `npu-smi` 一一绑定；前 8 卡不得参与。
- 耗时相对 CPU FP64 SOAP 大幅下降。
- 8 卡 NPU vs 8 卡 GPU 吞吐 1:1 或更好。
- 每个性能提交可独立验证、独立回退，且有证据包。

## Key Facts
| Item | Value |
|---|---|
| 权威分支 | `ascend_npu_optimize` |
| 远端最新核验 HEAD | `3a1d763`（仅将 SOAP 两处 MX QR 回退到 `torch.linalg.qr`） |
| 远端工作树 | `soap.py` 叠加未提交 stale-Q/per-factor Event 调度实验；当前运行不是纯 HEAD |
| 精度合同 | 63861df CPU FP64 双轴 SOAP；不能靠 `fb979b2` 快路径同时满足双门禁 |
| BAD 样本 | 本地 `step260_qr_bad_tensors/`；同事机 `/home/ubuntu/rank{0-7}_step10_ind0_192x192_BAD.pt` |

## STEP-265 / STEP-266：社区 QR NaN 与 SOAP 精度
两个独立问题：
1. **NaN**：`mx_driving_cloud.linalg.qr` 对有限 `[192,192]` 的最后 64 列（列 128–191）返回非有限 Q/R。A 全有限、无 0/denormal，`absmax≈7.91e-8`，`cond2≈1763`，8 rank 哈希相同。同 A 的 numpy CPU FP64/FP32 QR 成功；CPU 预处理后再走 mx QR 仍 NaN（STEP-257），CPU FP64 QR 无 NaN（STEP-258）。broadcast 只是用 rank0 正常 Q 盖住坏结果。
2. **精度 11/30**：与 QR 后端无关。mx QR + broadcast 与 CPU FP64 QR + broadcast 轨迹几乎相同（step30 约 +25%）。真正接近 GPU 的是 STEP-245 HEAD 610 行 SOAP（28/30 ≤1%）。

**Status:** complete_operator_root_cause_pinned_awaiting_user_path

## Decisions Made
| Decision | Rationale |
|---|---|
| 随机性移除单独成提交 | 隔离行为变更与性能变更 |
| 先证据后优化 | 禁止凭经验改热路径 |
| 精度合同冻结为 63861df SOAP | 2% 门禁下快路径全部不合格 |
| 禁止把 broadcast 当原始 SOAP | 用户已拒绝；broadcast 只掩盖 NaN |
| 禁止 overlay 抹掉 `fb979b2` | 正式修复必须新 commit |

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| 同事机账号 `ubantu` SSH 认证失败 | 1 | 实际用户是 `ubuntu`，家目录 `/home/ubuntu` |
| 跳板机无法直连同事公网机（Timeout） | 1 | 改本机先下载再直传 |
| 远端宿主无 `rg` | 1 | 改用限定目录的 `grep -R`，不在远端安装 |
| 本机缺少 GitHub CLI | 1 | 使用 winget 安装 GitHub CLI 2.97.0；当前尚未登录 GitHub |
| Gitleaks 首次调用未解析出可执行路径 | 1 | 安装本身成功；改用 `CommandInfo.Source` 或 `FileInfo.FullName` 分支解析后重跑，不重复原命令 |
| 暂存区 Gitleaks 扫描发现 8 个疑似敏感项 | 1 | 不输出匹配内容；仅提取文件、行号和规则 ID，逐项脱敏或排除后重扫 |
| PowerShell 脱敏元数据提取脚本出现空管道语法错误 | 1 | 改为先把 `foreach` 结果赋给数组，再单独格式化输出 |
| 首个 `wandb_key` 精确引号正则未匹配原行结构 | 1 | 不读取原值；改为整行锚定 `wandb_key = ...` 后替换为固定空值注释 |
| `wandb_key` 脱敏替换第二次仍未匹配 | 2 | 停止重试并派发只读故障召回；检查发现 PowerShell 单引号正则中把 `\s` 误写成了双反斜杠字面量，后续改用正确的 `\s` 正则语义 |
| 脱敏脚本错误消息中的 PowerShell 变量后紧跟冒号导致解析失败 | 1 | 用 `${rel}` 明确变量边界后重跑；脚本在文件读写前即停止，无内容变更 |
| GitHub 连接器首次读取仓库元数据发生传输错误 | 1 | 暂不重试连接器；本地提交后优先用现有 Git 远端推送，再用连接器或 CLI 创建 PR |
| `git diff --cached --check` 报历史材料中的尾随空白 | 1 | 不批量改写补丁/历史源快照以免改变语义；Python AST、Gitleaks、范围及文件类型门禁作为本次归档验证 |
| Git 提交因缺少作者身份配置而停止 | 1 | 从仓库上一条提交读取作者名和邮箱并仅写入仓库本地配置，不展示或写入项目文件 |
| 记录首批提交状态的 `apply_patch` hunk 格式错误 | 1 | 修正补丁分段后重新应用；首次失败未修改文件 |
| `main` 首次纯快进被本地 `remote_exec.py` 与未跟踪 STEP-307 文件阻止 | 1 | 未覆盖本地内容；先形成独立本地增量提交，再用 `-X ours` 无冲突合并发布分支 |
| IPv4 文件扫描首次未启用 PCRE2，`rg` 不支持 look-around | 1 | 立即用 `rg --pcre2` 重扫，并按回环/非回环分类；仅发现回环地址，无远端地址硬编码 |
| 合并树全量 IPv4 检查发现两份历史 Markdown 各有 1 个非回环地址 | 1 | 未显示地址；机械替换为 `[REDACTED_IP]` 并重跑全部安全门禁 |

## Notes
- 历史 STEP-001～264、拒绝矩阵、profile 数字、A/B 表全部在 `planning_backup/2026-08-18/`。
- STEP-274 已完成；bypass 仍是容器内 patch，未 commit。

### Phase 18: STEP358 单次发布数学门禁远程编排器
- [ ] 对照 STEP357 远程构建器和最新三个 STEP358 运行脚本固化合同
- [ ] 实现 `.codex-tools/step358_run_release_math_remote.py` 及 `--dry-run`
- [ ] 静态验证、`py_compile` 和 Python 代码审查
- [ ] 把本地动作和未执行远程事实记入 `操作步骤.md`
- **Status:** in_progress

### Phase 19: STEP362 动态 simplifiedKey 候选重构
- [x] descriptor 缩减为同版 OPC simplified-key 白名单字段
- [x] 动态 `_0_mix_aic/aiv` 产物身份与 package 门禁实现、审核
- [x] 唯一 STEP362 隔离 OPC 构建与新 wheel 封包
- [x] 远端原位复核 manifest/artifact dynamic simplifiedKey 与 concrete identity
- [x] 后8卡8-rank 最小 candidate identity：8/8 candidate `_0_mix_aic`、original=0
- [x] 首个真实192×192输入：shape PASS、finite FAIL，已停止扩大测试
- **Status:** completed_candidate_runtime_invalid

### Phase 20: QrV2 剩余非有限根因二次源码定界
- [x] 复核修复候选的 LocalTensor/Matmul/DataCopy 异步生命周期
- [ ] 复核192×192最后panel zero-work、tiling字段与局部buffer初始化
- [x] 以新假设设计 v3 最小消融：仅补 `CalcQForLARFB` 的 MTE3→MTE2 依赖
- [x] v3→v2 SHA、事件位置/数量、SyncAll/UpdateA/base slot 不变的本地门禁
- [x] 独立审核 v3 并关闭 failure JSON/I/O 证据链 P1 与交付命名/占位 P2
- [x] 新建唯一隔离目录执行 OPC 构建；构建前控制器保持 `V3_RELEASE_READY=False`
- [x] 后8卡8-rank真实192输入已证明 v3 concrete AIC 命中，但 finite 失败，候选被设备证伪
- **Status:** completed_candidate_runtime_invalid

### Phase 21: QrV2 v4 同步/所有权闭合候选
- [x] 以 v3 设备 finite 失败为边界，冻结 v4 仅含三项源码增量
- [x] `CalcQForLARFB` 每核 scratch 与 `coreId < blockp`/workspace 范围门禁
- [x] `UpdateAForLARFB` 独立 MTE3→MTE2 依赖门禁
- [x] core0 Q 写回后全核 `SyncAll`、再释放 T/V/A 的顺序门禁
- [x] 三项精确反向恢复 v3 SHA，拒绝 shared scratch/漏事件/barrier 错位负例
- [x] 发布身份、shadow、worker 与未构建控制器升级为 v4；本地测试与 Python 审查
- [x] 独立审核 P0/P1（P0=0、P1=0；P2=1 非阻断测试可维护性建议）
- [x] 唯一 STEP370 隔离 OPC 构建与封包，installed/runtime inventory 前后闭合
- [x] 独立复核 STEP370 发布包 manifest/RECORD/双 SoC/concrete identity 结构
- [x] 关闭 STEP371 launcher ownership/identity 失败路径统一 cleanup/postflight P1，故障注入与独立复审通过
- [x] 后8卡8-rank首个真实 192×192 identity/finite 门禁已执行；v4 AIC 8/8命中但 finite FAIL
- [x] 以 v4 与 v3 相同失败计数为线索重新审计源码，已形成 v5 静态可证伪单变量候选；后续由 Phase22 接管
- [x] 新候选的 CPU 官方 QR 语义、状态序列、30-step loss≤2%和耗时门禁已设计，执行与裁决由 Phase22 接管
- [x] 修正 loss gate：GPU=0/NPU=0 PASS、GPU=0/NPU≠0 FAIL，0/0 与 0/nonzero 测试通过
- **Status:** v4_concrete_runtime_finite_failed_source_reaudit

### Phase 22: QrV2 v5 Matmul tensor-position 合同修复
- [x] 用 v4 真机计数将首次污染定界到首次 LARFB：`18528-2080=16448`
- [x] 独立审计全部 Matmul 声明与 `SetTensorA/SetTensorB/IterateAll` 实参位置
- [x] 生成 v5 最小候选：CalcQ 第二乘直接使用 `vLocal(VECIN)`，`qa=VECIN/GM/VECIN`
- [x] 精确反向两项修复恢复 v4 SHA，7/7 正反例、`py_compile`、patcher `--check`、新 Python diff-check 和 STEP372 文档区段行尾检查通过
- [x] 将已确认事实、根因候选、未验证项、分级门禁、失败分支和回滚方式写入实施方案
- [x] 将 builder/shadow/worker/controller 全部锁定唯一 `QrV2_matmul_position_fix_v5`；设备 controller 在新 wheel 审计前保持 `V5_RELEASE_READY=False`，显式禁止 v4 wheel 路径/SHA
- [x] 在唯一 STEP373 隔离目录完成目标 CANN 8.3 OPC 构建/封包，wheel/RECORD/双 SoC/concrete entry/binary-info/installed inventory 及独立原位审计全部闭合
- [x] 唯一尝试启动后8卡、8-rank、每 rank 一个真实192输入的 v5 核心门禁；停止在runtime timeout/trap，profiler identity与全部数学谓词均未评估
- [x] 审计 STEP374 的 controller_error、ready/done/failure、profiler identity 与清场证据：8/8 v5 AIC timeout/trap，0 done，无Q/R数值，清场闭合
- [x] 不运行NPU完成PC/ELF映射与两项delta源码审核；PC只能到AIC/AIV函数offset，delta2 lowering为活跃嫌疑但尚非根因
- [x] 实现delta1-only本地诊断生成器：相对v5只撤回delta2，独立identity/SHA、四格SHA和反向v4门禁闭合，明确非发布候选
- [x] 独立代码复审delta1-only生成器；两轮P2加固后最终P0/P1/P2=0
- [x] 实现该单变量probe的独立diagnostic adapter与测试；base package/all旁路、approved_root、原子seal和产物闭包均已加固，复审P0/P1=0
- [x] 补齐alias分叉与seal前篡改两个P2定向负例，锁定最终测试SHA
- [x] 实现专用远端controller并完成首次phase-transition；唯一执行在exclusive目录shell语法层失败，未创建目录/上传/OPC，已回退`BUILD_READY=False`且未重试
- [x] 对修复后的controller重新独立审核并执行唯一retry2；在首个before-snapshot因官方OPC alias为symlink被拒，未启动OPC且未重试
- [x] 修复controller统一使用合同中已解析的OPC realpath，回退`BUILD_READY=False`并完成独立审核P0/P1/P2=0；base/adapter symlink门禁未放宽
- [x] 以全新attempt3目录完成本地武装并执行唯一OPC构建；状态`diagnostic_built_unvalidated`
- [x] 远端原位重算双SoC产物/identity/SHA/诊断标志/库存/进程闭包；未下载、未打包、未安装、未运行NPU
- [x] 设计不打包、不改installed的diagnostic shadow与后8卡一次性NPU诊断加载链；明确复用/禁止边界和门禁
- [x] 实现独立diagnostic shadow builder与负例；三轮审核关闭真实policy、ZIP/tree、dirfd/事务/fd泄漏问题，终审P0/P1/P2=0
- [x] 实现首个真实192 diagnostic worker薄适配器并终审P0/P1/P2=0
- [x] 实现diagnostic host adapter并终审P0/P1/P2=0
- [x] 完成专用remote controller真实容器动作、上传闭包、rank/launcher ownership与pidfd timeout清场；历次P0/P1均已关闭，两路终审P0=0/P1=0；保持`NPU_READY=False`等待phase-transition
- [ ] 独立审核后另行申请一次后8卡NPU诊断；结果不得升格为修复
- [ ] 根据诊断证据另行设计正式修复候选；以全新identity/SHA重走构建、首输入profiler identity+数学门禁
- [ ] 只有新候选首输入通过后，才执行 CPU 官方 QR shape/padding/语义、状态序列和训练调用账本门禁
- [ ] 只有上述门禁通过后才执行 30-step：每步 loss 相对 GPU log `<=2%`，missing/duplicate/nonfinite 任一失败
- [ ] 只有上述门禁通过且测试前锁定耗时最大允许偏差，才比较单算子 median/P95 和训练 Iter2–30/耗时周期；不允许事后放宽
- **Status:** step377_full_chain_reviewed_p0_0_p1_0_phase_transition_pending

### Phase 23: delta2-only 端到端训练验证

- [x] STEP392 standalone 后8卡 world8：8/8 candidate identity、finite和QR数学PASS
- [x] 校正结论：历史原算子standalone也可PASS，STEP392不能证明训练NaN已修复
- [x] 暂停release promotion和非关键cleanup修正，不进入训练前的重复单算子测试
- [x] 锁定与GPU loss oracle/STEP204合同一致、仍含两处MX QR的唯一commit和工件SHA
- [x] 实现低扰动delta2 shadow 30-step训练controller：后8卡、8rank、`mapqr-leicheng`，不逐QR dump/profile/sync；本地最终审核P0/P1=0
- [ ] 唯一低扰动30-step：训练内NaN/finite、30/30 loss偏差≤2%、Iter2–30耗时门禁
- [ ] 若低扰动训练失败，只对确定step/call做定向训练内抓取；若通过，再做首个真实QR的最小identity profile
- **Status:** step393_attempt6_torch_npu_import_failure_handoff
