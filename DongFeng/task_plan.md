# Task Plan: Ascend NPU 性能优化

历史全文已备份到 `planning_backup/2026-08-18/`。本文件只保留当前有效目标、门禁和下一步。

## Goal
在不改变最终功能、训练语义且保持 loss/梯度门禁的前提下，为 `ascend_npu_optimize` 做可复现、可量化、按功能独立提交的昇腾 NPU 性能优化。验收：同合同下 8 卡 NPU 与 8 卡 GPU 的 `throughput (samples/s)` 达到 1:1 或更好。用户 2026-08-18 双门禁：耗时相对 CPU FP64 SOAP 基线大幅下降，且逐步 logged `loss` 相对 GPU `|Δ| ≤ 2%`。

## Next Step
验证并发布上一轮远端核验时新出现的 STEP-307 源码脚本，再确认项目源码无遗漏。

## Current Phase
**STEP-303 DONE**：同一 BAD tensor 在 8 卡可见且正确 `set_device` 的逻辑卡高频重放中也未复现 NaN/507015；单算子脚本侧的 device-context 问题已基本澄清，剩余重点转向真实训练链路中的上下文与状态因素。

## Standing Rules
- 所有远端访问必须锁定 `机器IP.md` 中地址末段为 `42` 的主机器，并在第二跳连接后核验主机身份；不匹配立即停止。
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
- [ ] 验证并提交 STEP-307 源码脚本，推送后复核
- **Status:** in_progress

## Acceptance Criteria
- 逐步 logged `loss` 相对 GPU `|Δ| ≤ 2%`。
- 耗时相对 CPU FP64 SOAP 大幅下降。
- 8 卡 NPU vs 8 卡 GPU 吞吐 1:1 或更好。
- 每个性能提交可独立验证、独立回退，且有证据包。

## Key Facts
| Item | Value |
|---|---|
| 权威分支 | `ascend_npu_optimize` |
| 上次核验 HEAD | `9565044`（`669a138` SOAP + 仅 `mx_driving_cloud.linalg.qr`） |
| 工作树 | SOAP 已入库；eval/config/`loading.py`/`run_*.sh` 仍有无关脏文件，未进本次 commit |
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

## Notes
- 历史 STEP-001～264、拒绝矩阵、profile 数字、A/B 表全部在 `planning_backup/2026-08-18/`。
- STEP-274 已完成；bypass 仍是容器内 patch，未 commit。
