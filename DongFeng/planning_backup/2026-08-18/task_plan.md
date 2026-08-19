# Task Plan: Ascend NPU 性能优化

## Goal
在不改变最终功能、训练语义且保持 loss/梯度门禁的前提下，为 `ascend_npu_optimize` 分支执行可复现、可量化、按功能独立提交的昇腾 NPU 性能优化；最终验收目标为同合同下 8 卡 NPU 与 8 卡 GPU 的 `throughput (samples/s)` 比达到 1:1 或更好。用户 2026-08-18 明确双门禁：耗时相对 CPU FP64 SOAP 基线大幅下降，且逐步 logged `loss` 相对 GPU `|Δ| ≤ 2%`（旧 1% 门禁放宽为此）。

## Next Step
社区 QR 的 NaN 与 SOAP 精度漂移已拆开。等用户选择：恢复 STEP-245 HEAD SOAP 再只换 QR，或把 8 个 BAD `[192,192]` `.pt` 交给算子同事修 kernel。禁止把 broadcast 当原始 SOAP 逻辑。

## Current Phase
**STEP-265 ROOT-CAUSE SPLIT**：NaN = 社区 QR 对部分 192×192 吐非有限 Q/R；11/30 精度 = 63861df 风格 SOAP 工作树，不是 mx QR 独有。

## Plan Status
**APPROVED / RESUMED ON NEW MACHINE — 2026-08-12**

## Approval Gate
- 当前仅允许：只读检查、社区/官方资料研究、编写 `task_plan.md`、`findings.md`、`progress.md`。
- 用户明确批准本计划前，禁止修改业务代码、配置或脚本，禁止提交、推送，禁止运行会改变项目或远程环境状态的命令。
- 用户批准后仍按阶段执行；超出已批准范围的变更必须再次确认。

## Performance Evidence Gate
- 除非能从代码与算子语义直接证明存在性能问题，否则禁止仅凭经验、关键词或“可能更快”实施优化。
- 明确问题仅包括可直接确认的热路径 CPU fallback、每 step 强制 D2H/H2D、显式同步、逐元素 host 同步、已知调试模式常开等；即使属于明确问题，修改后仍必须进行工具化 A/B 验证。
- 其他所有候选必须先用实际训练或可代表真实调用链的测试采集证据，再决定是否修改。
- 默认证据链：普通基线计时 -> `torch_npu.profiler` 采集 -> `msprof-analyze advisor`/timeline 定位 -> 必要时定向微基准 -> 单变量修改 -> 同口径复测与 profile 对比。
- 静态工具只负责生成候选，不能单独证明运行时瓶颈；`msfmktransplt` 报告必须与 profiler、fallback 日志或定向基准交叉确认。
- 没有“问题证据 + 优化后收益证据 + 正确性证据”的候选，不得形成 `【npu性能优化】` 提交。

## Training Command Source
- 训练、基线、profile 和回归命令必须参考远程项目当前目录下已存在的 `.sh` 脚本，不自行假设或重新拼造训练入口。
- 所有正式训练均使用 8 卡，包括正确性检查、性能基线、profiler 采集、优化后 A/B 和最终回归。
- 8 卡的实际设备编号、`world_size`、HCCL/launcher 参数及绑核方式以远程已有 `.sh` 脚本为准，不自行假设为某组卡号。
- Phase 1 先只读盘点所有相关 `.sh`：实际调用的 Python 入口、配置文件、卡号/卡数、launcher、环境变量、数据/权重、work-dir、resume、日志重定向和后台运行方式。
- 选择与当前目标分支及目标训练场景匹配的脚本作为 canonical command，并记录脚本路径与文件 hash。
- 测试需要增加 profiler、短迭代或输出目录参数时，优先使用脚本已有透传能力或命令行覆盖；若必须修改/新增测试脚本，需在计划获批后进行，并与业务优化提交隔离。
- 基线与优化后复测必须使用同一脚本、同一参数和同一环境；任何必要差异都要显式记录。

## Phases

### Phase 1: 现状审计与计划定稿
- [ ] 读取 `AGENTS.md` 与 `机器IP.md`，确认项目和远程机器规则
- [ ] 确认本地/远程仓库位置、当前分支、工作区状态和 `asend_npu_optimize` 分支状态
- [ ] 审查 loss 对齐相关提交及其改动意图
- [ ] 全量盘点随机种子、确定性开关、CPU fallback、设备搬运和非 NPU 亲和算子
- [ ] 识别训练/评测入口、数据、环境和现有测试能力
- [ ] 枚举并只读解析远程当前目录已有训练 `.sh` 脚本，确定 canonical 训练/profile 命令
- [ ] 确认现有脚本的 8 卡设备映射、rank/world size、HCCL 配置和每卡 batch size
- [ ] 调研官方/社区成熟的昇腾亲和算子与性能采集工具
- [ ] 给出候选优化清单、风险、优先级、验证方法和提交边界
- [ ] 将完整计划提交用户审批
- **Status:** in_progress

### Phase 2: 移除固定随机性并独立提交
- [ ] 先用历史 diff 建立“loss 对齐阶段新增随机性代码”白名单，不按关键词盲删
- [ ] 分类处理：全局硬编码 seed、确定性开关、调试环境变量、DataLoader seed、sampler 内 seed
- [ ] 删除此前为 loss 对齐加入的固定随机性，恢复框架/项目原有随机行为
- [ ] 保留或重构分布式正确性必需的 rank 间共享随机协议，并记录理由
- [ ] 清除仅服务于随机性对齐的调试日志，避免性能和日志污染
- [ ] 运行语法检查、最小数据构建和 8 卡短跑，并检查跨 rank sampler 一致性
- [ ] 确认差异不含任何算子或性能优化改动
- [ ] 创建独立提交：`【loss对齐】随机性移除`
- [ ] 若该提交已在先前会话产生，先核对 SHA、完整 diff、父提交、工作区状态和验证结果；未经用户批准不重写历史
- **Status:** pending

### Phase 3: 基线环境冻结与基线测试
- [ ] 记录硬件、驱动、固件、CANN、PyTorch、torch_npu、依赖和关键环境变量
- [ ] 以“随机性移除”提交作为性能基线代码版本
- [ ] 固定测试命令、数据范围、权重、每卡/全局 batch size、8 卡设备映射、精度模式、预热策略和迭代数
- [ ] 基于远程已有 `.sh` 脚本执行基线，并保存脚本路径、hash、展开后的非敏感参数和环境摘要
- [ ] 完成正确性基线：loss、关键输出、异常与 fallback 日志
- [ ] 完成性能基线：吞吐、step time、端到端耗时、显存、主机占用、编译/预热开销
- [ ] 每种基线至少 3 次重复运行，记录中位数、P95 和离散度；首轮编译与稳态分开统计
- [ ] 在 8 卡训练下使用 Ascend PyTorch Profiler 采集 timeline、operator、memory、communication 数据并保存各 rank 原始产物
- [ ] 使用 `msprof-analyze advisor all` 生成自动诊断报告；版本支持时增加 `module_statistic`
- [ ] 使用 `npu-smi`/系统工具按固定采样周期记录设备与主机资源利用率
- [ ] 将基线结果写入 `progress.md`，形成后续统一对比表
- **Status:** pending

### Phase 4: CPU fallback 与非亲和算子替换
- [ ] 以 profiler、fallback 日志和静态审计交叉确认真实热点
- [ ] 将候选分为“静态可明确确认”和“必须工具定位”两类，并记录分类依据
- [ ] 对非明确候选先采集 operator self/total time、调用次数、AICPU、host/device 空洞、同步和内存证据
- [ ] 对每个候选算子先检索 Ascend/PyTorch/torch_npu/社区成熟实现与版本约束
- [ ] 按“官方内置 NPU 算子 > 官方融合算子 > 活跃社区成熟实现 > 必要时自定义算子”选择方案
- [ ] 分批替换 CPU fallback、隐式 CPU 同步、频繁 H2D/D2H、低效组合算子
- [ ] 每批变更进行正确性、稳定性和性能 A/B 验证；无收益或有回归则回退该候选
- [ ] 每个逻辑独立优化使用提交前缀：`【npu性能优化】xxxxx`
- **Status:** pending

### Phase 5: 系统级与训练链路优化
- [ ] 分析数据加载、host-device 同步、动态图编译、内存格式、混合精度和通信开销
- [ ] 仅对 profiler 证实的瓶颈实施优化
- [ ] 检查算子融合、异步执行、梯度处理、优化器及分布式通信的成熟 NPU 方案
- [ ] 每项优化执行相同口径的 A/B 测试并独立提交
- **Status:** pending

### Phase 6: 回归验证与结果交付
- [ ] 使用统一测试矩阵进行正确性、性能、稳定性和资源占用回归
- [ ] 对比基线与每项优化的收益、波动、兼容性和风险
- [ ] 核对提交历史、提交前缀和变更边界
- [ ] 汇总最终优化清单、未采用方案及原因、复现命令和性能报告
- **Status:** pending

### Phase 7: 新机器迁移恢复与基线重建
- [ ] 读取本地机器连接信息但不输出凭据，确认新机器连接链路
- [ ] 只读确认目标仓库路径、`ascend_npu_optimize` 分支、HEAD、工作树和既有两项优化提交
- [ ] 只读确认完整名称为 `mapqr-leicheng` 的容器存在且运行状态正常
- [ ] 在该容器内核验 Python、PyTorch、`torch_npu`、CANN/HCCL 与 profiler 工具，不改变任何版本
- [ ] 使用 `npu-smi` 核验 8 张物理 NPU 的健康、内存和现有进程，确认无资源冲突
- [ ] 解析仓库现有 `.sh` 训练入口、8 rank/device 映射、数据/权重/输出目录与恢复参数
- [ ] 对照迁移清单核验代码、配置、权重和必要运行目录是否齐备；不读取或下载数据内容
- [ ] 冻结新机器 canonical command、代码/config hash 与脱敏环境摘要
- [ ] 在 `mapqr-leicheng` 内执行一次 8 NPU 最小恢复验证，启动后核验 `torch_npu`、8 rank 和 `npu-smi` 进程
- [ ] 完成新机器三次普通基线与一次代表性 profiler，远端原位分析并形成新基线报告
- **Status:** in_progress

## Acceptance Criteria
- 固定随机性代码完整移除，且仅存在于独立提交 `【loss对齐】随机性移除`。
- 已识别的 CPU fallback 均完成处置：替换为成熟 NPU 亲和方案，或记录不可替换的证据与影响。
- 每个性能优化提交均使用 `【npu性能优化】` 前缀，且提交粒度可独立验证和回退。
- 有版本、环境、命令、数据口径完整的基线记录和优化后对比结果。
- 所有保留优化均通过正确性验证，并给出可复现的性能收益。
- 每个性能提交均关联完整证据包；除静态可明确问题外，必须包含修改前的工具定位结果。

## Read-only Audit Commands and Outputs
- Git：`status --short --branch`、`branch -avv`、loss 相关 `log --all --grep`、候选提交 `show --stat/--name-status/--format=fuller`、父提交对比。
- 随机性静态扫描：`seed`、`manual_seed`、`Generator`、`deterministic`、`PYTHONHASHSEED`、NPU/HCCL 确定性变量、KMeans `random_state`。
- fallback 静态扫描：`.cpu()`、`.numpy()`、`.tolist()`、`.item()`、CPU tensor 创建、NumPy/SciPy/Scikit-learn/OpenCV 热路径、显式同步、H2D/D2H。
- NPU 配置扫描：compile mode、internal format、HF32、AMP、融合优化器、同步开关、调试/dump/profile 环境变量。
- 训练脚本扫描：当前目录及必要子目录中的 `.sh`，解析其 Python 入口、config、launcher、设备选择、环境初始化、后台执行与日志路径。
- 输出：文件/行号/调用链/执行频率假设/正确性风险/成熟替代方案/是否需 profile 验证的审计表。

## Performance Toolchain
| Layer | Tool | Purpose | Planned Output |
|---|---|---|---|
| 静态迁移审计 | `mstt/msfmktransplt` PyTorch Analyse | API 支持、亲和 API、动态 shape、性能建议 | CSV/TXT 报告 |
| 框架与算子采集 | `torch_npu.profiler` | PyTorch/CANN/NPU operator、内存、通信、timeline | `ascend_pt`、JSON/CSV/DB |
| 自动诊断 | `msprof-analyze advisor` | AICPU、动态 shape、融合、亲和 API、下发、同步、通信问题 | HTML/XLSX/终端摘要 |
| 前后对比 | `msprof-analyze compare` | baseline 与候选优化 profile 对比 | 对比报告 |
| 设备监控 | `npu-smi` 与系统采样 | 利用率、显存、功耗/频率（环境支持时）、CPU/IO | 时间序列日志 |
| 可视化 | MindStudio Insight 或 Chrome trace | 人工查看 host/device 空洞和同步链 | 截图/结论摘要 |

## Initial Optimization Backlog
| Priority | Candidate | Evidence Needed | Preferred Mature Direction | Main Risk |
|---|---|---|---|---|
| P0 | SeTa 每 step 的 NPU -> CPU loss/index 搬运 | timeline 中 D2H/同步占比 | 批量化、降低更新频率，或保持状态在 NPU 后仅阶段性汇总 | sampler 语义变化 |
| P0 | Scikit-learn KMeans + `.numpy()` | 调用频率与 host 耗时 | 优先 TorchNPU 原生张量方案/成熟 NPU 聚类实现；无成熟方案则先减少频率 | 聚类结果与 loss 轨迹变化 |
| P0 | `.tolist()` / Python `.item()` 循环 | host self time、同步事件 | 张量化分组、排序、聚合，避免逐元素同步 | 索引/排序稳定性 |
| P0 | AICPU/CPU fallback 算子 | profiler AICPU 与 fallback 日志 | 按版本查 TorchNPU 原生/op-plugin/亲和 API | 算子语义和 dtype 差异 |
| P1 | `allow_internal_format=False` | A/B profile、精度检查 | 恢复默认/开启内部格式，按热点算子验证 | 布局改变引起精度/兼容问题 |
| P1 | `jit_compile=False` 与动态 shape | 编译次数、稳态性能 | 按当前 torch_npu 版本评估图编译/动态 shape 策略 | 首轮编译时间或不支持图 |
| P1 | HF32 全关闭 | MatMul/Conv 占比与精度容限 | 分别 A/B `conv.allow_hf32`、`matmul.allow_hf32` | loss 漂移 |
| P1 | 原生 AdamW/梯度裁剪 | optimizer 时间占比 | `NpuFusedAdamW` 与 fused grad norm（版本支持时） | checkpoint/数值行为 |
| P1 | 重复 permute/reshape/cast/contiguous | 融合建议与算子序列 | 官方融合 API（如版本匹配的 confusion transpose） | shape/layout 约束 |
| P2 | DataLoader、pin memory、worker 数 | DataLoader gap、CPU/IO 利用率 | 参数扫描、预取/持久 worker（框架版本支持时） | 主机内存和数据顺序 |
| P2 | HCCL 通信与同步 | communication timeline | bucket、overlap、通信数据对齐等官方建议 | 多卡稳定性 |

## Baseline and A/B Protocol
1. 记录不可变标识：commit SHA、远程训练脚本路径及 hash、配置文件 hash、权重、数据清单/范围、容器或环境版本。
2. 所有训练测试统一使用 8 卡；单算子微基准如需减少卡数只能用于算子机制验证，不能作为训练性能结论。
3. 区分 cold start、warmup、steady state；正式计时窗口内禁止开启额外 debug 输出。
4. 每个候选只改变一个逻辑因素，至少重复 3 次；报告 median、P95、吞吐、显存和波动。
5. profiler 有采集开销，普通计时结果与 profiler 结果分开记录；profile 用于归因，不直接当吞吐结论。
6. 正确性先于性能：与基线比较 loss 趋势、关键张量/指标、NaN/Inf、样本覆盖和多卡一致性。
7. 保留门槛：收益超过测量噪声且正确性通过；否则不提交或回退该项，并记录未采用原因。
8. 非明确问题不得跳过修改前 profile；明确问题允许直接进入定向 A/B，但不得跳过修改后验证。
9. 8 卡结果同时记录全局吞吐和各 rank step time/显存；检查慢 rank、通信等待和卡间离散度。

## Required Evidence Package per Optimization
| Evidence | Required Content |
|---|---|
| Candidate classification | 静态明确问题 / 工具定位问题，以及分类理由 |
| Before measurement | commit、命令、数据、8 卡映射、world size、每卡/全局 batch、迭代窗口、至少 3 次计时结果 |
| Command provenance | 远程已有 `.sh` 脚本路径、hash、实际入口和非敏感参数摘要 |
| Bottleneck proof | profiler 文件、advisor/timeline 结论、算子调用次数与耗时；明确问题可用代码路径证据替代修改前 profile |
| Replacement source | 官方文档、TorchNPU/OpPlugin 版本或成熟社区实现链接 |
| Correctness | loss/关键输出/NaN-Inf/多卡行为对比 |
| After measurement | 同环境、同命令、同数据的至少 3 次复测和波动 |
| Decision | 保留、回退或暂缓，以及理由 |

## Commit Strategy
| Order | Commit | Allowed Scope |
|---:|---|---|
| 1 | `【loss对齐】随机性移除` | 仅随机种子、确定性固定及其专用调试代码 |
| 2+ | `【npu性能优化】<具体对象与动作>` | 每次一个可独立 A/B、可独立回退的优化逻辑 |

- 禁止把 baseline 产物、大型 profile 数据、训练日志、凭据或临时文件误提交。
- 每个性能提交说明：瓶颈证据、成熟方案来源、变更范围、正确性结果、性能前后数据、适用版本。
- 未经用户明确要求不 push、不创建 PR、不改写已共享历史。
- 功能原子性规则：同一个算子优化或同一个完整功能只形成一个 commit，必要的代码、配置和直接相关测试一起提交；禁止把同一功能拆成准备、实现、修补等多个零碎提交。
- 仅当功能复杂且各子能力均可独立验证、独立回退时才允许拆分；拆分前在计划与 `操作步骤.md` 中明确每个 commit 的功能边界。

## Stop / Escalation Conditions
- 工作区存在无法归属的用户改动或当前分支不符：停止并报告。
- 当前 CANN/torch_npu 版本无成熟替代或社区方案版本不匹配：不强行替换，提供选项。
- 正确性超出既定容差、训练不稳定、性能收益低于噪声：停止该候选并回退。
- 基线数据/权重/运行命令不完整，或资源被其他任务显著干扰：不形成性能结论。
- 调查发现范围明显扩大（需升级框架、CANN、容器或改数据语义）：另行请求批准。

## Key Questions
1. loss 对齐阶段具体引入了哪些随机性固定、CPU fallback 或实现折中？
2. 哪些候选算子在实际 NPU profile 中构成热点，而非仅静态可疑？
3. 当前 CANN/torch_npu 版本支持哪些官方亲和或融合算子？
4. 项目可接受的正确性误差、性能波动和测试时长分别是多少？
5. 基线运行所需数据、权重、命令和资源是否已齐备？

## STEP-265：社区 QR / SOAP 精度因果拆分（2026-08-18）

- 用户问：结合实际测试，能否知道原因。
- 结论：**两个独立问题，不能用一个原因解释。**
  1. **NaN（STEP-255/257/259/260）**：`mx_driving_cloud.linalg.qr` 对部分有限 `[192,192]` 输入直接返回非有限 `Q/R`。STEP-260：4408 次中 4400 次 `Q@R≈A`（max 7.2e-6），**8 次 BAD、每 rank 各 1 次**。开 broadcast 后 NaN 消失，因为坏的 Q 被 rank0 正常结果覆盖。
  2. **精度 11/30 ≤2%（STEP-254/256/258/259）**：与 QR 后端无关。mx QR + broadcast 与 CPU FP64 QR + broadcast 轨迹几乎相同（step30 约 +25%）。真正接近 GPU 的是 STEP-245 的 HEAD 610 行 SOAP（28/30 ≤1%）。当前工作树是 63861df 风格还原 + mx QR，缺 foreach/wave 结构。
- 样本已在本地 `step260_qr_bad_tensors/`，并传到同事机 `/home/ubuntu/`。
- **Status:** complete_root_cause_split_awaiting_user_choice

## Decisions Made
| Decision | Rationale |
|---|---|
| 计划批准前设置硬门禁 | 用户明确要求不得直接修改代码 |
| 随机性移除单独成提交 | 遵守指定提交信息并隔离行为变更与性能变更 |
| 先基线/profile、后优化 | 避免凭静态印象优化，确保收益可量化 |
| 算子替换优先官方和成熟社区实现 | 降低正确性、维护性与版本兼容风险 |
| 每项性能优化独立验证和提交 | 便于定位收益、回归和回退 |

## Errors Encountered
| 2026-08-14 STEP-174 | 首次容器schema验证用`docker exec`未加`-i`，here-doc未传入且没有schema输出；架构10节检查独立通过 | 加`docker exec -i`重跑，schema 0错误、20/20 bubble有host context和项目栈 |
| 2026-08-14 STEP-174 | 分析器传入诊断目录覆盖config时，其相对`_base_`被按诊断目录解析，`config_summary`为FileNotFound | 改用仓库内实际base config绝对路径重跑分析器，只修复架构/config元数据，不改变profile或bubble算法 |
| 2026-08-14 STEP-174 | 合成测试首条命令含递归删除旧output，被本地安全策略拒绝；未执行分析器 | output目录原本不存在，移除删除动作后重跑 |
| 2026-08-14 STEP-174 | 合成测试缺本地`ijson`，分析器在import阶段退出，随后断言文件不存在 | 按本地安装授权装入项目专用`.codex-tools/python-packages`，不改远端客户环境；重跑时显式设置PYTHONPATH |
| 2026-08-14 STEP-174 | 首次误在`.codex-tools/schema.json`读取schema，文件不存在 | 改读技能正式路径`ascend-profiling-anomaly/references/schema.json`；仅本地只读失败，无远端状态变化 |
| Error | Attempt | Resolution |
|---|---:|---|
| 暂无 | 0 | - |
| R6夹具哈希命令的通配符包含`__pycache__`目录，`sha256sum`返回1并使后半门禁提前退出 | 1 | 改用`find -maxdepth 1 -type f -exec sha256sum`，不重复通配符方案；训练尚未启动 |
| R6首次截取npu-smi后8卡区段的`sed`地址表达式解析失败 | 1 | 放弃区段地址，改用精确行正则筛选NPU 4～7，并通过宿主`/proc/PID/status:NSpid`建立宿主PID到容器rank PID映射；训练不受影响 |
| R6初始化期状态查询包含容器内全进程扫描，超过本地30秒命令时限 | 1 | 训练未被操作；后续高负载期只读日志、宿主launcher PID和端口，不重复容器内全进程扫描 |
| STEP-178远端宿主未安装`rg`，首次源码定位命令失败 | 1 | 远端环境禁止安装依赖；改用限定目录/扩展名的`grep -R`，不重复`rg`方案 |
| STEP-178诊断config的`_base_`相对诊断目录解析，首次指向不存在的`test_harness/projects` | 1 | 训练未启动；把base改为从diagnostics兄弟目录回到业务仓库的明确相对路径，再做Config门禁 |
| STEP-178更新夹具SHA时错误复用了会匹配`__pycache__`的通配符 | 1 | 训练未启动；立即改为`find -maxdepth 1 -type f -exec sha256sum`，并将此已知错误再次写入操作记录避免后续复用 |
| 同事机账号 `ubantu` SSH 认证失败 | 1 | 实际用户是 `ubuntu`，家目录 `/home/ubuntu`；8 个 BAD `.pt` 已直传成功 |
| 跳板机无法直连同事公网机（Timeout） | 1 | 改本机先下载再直传；用户一度取消上传后又明确要求本机直传 |
| STEP-178生产patch按原文件CRLF写入新行，`git diff --check`把新增行的CR判为trailing whitespace | 1 | 正确性门禁未启动；恢复目标文件，改为仅让新增块使用LF而保持未改行原字节，重新应用并复核最小diff |

## Notes
- 远程机器信息必须从本地 `机器IP.md` 读取，不在规划文件或回复中复制凭据。
- 本计划在 Phase 1 调查完成后会细化为文件/算子级候选清单；仍需用户明确批准才能实施。
- 当前执行候选：将 `fix_pts_backbone && !pts_bbox_head` 分支内的全模型 `self.eval()/self.train()` 收敛到 `self.pts_backbone.eval()/train()`；只允许一个功能 commit。静态门槛、三次 8-NPU 30-step 性能复测和 loss/grad 有限性均通过才保留，否则精确回退。
- 候选结论：冻结点云骨干局部模式切换正确性通过但性能仅 +0.8897%，低于噪声门槛，已回退、无 commit。下一步重新筛选 normal-step 中尚未验证且能保持数值轨迹的候选。
- 下一候选筛选：梯度裁剪参数缓存因理论上限约 1.6% 已拒绝；当前只读提取 SOAP contraction 的真实 shape/调用族，只有在能保持参数顺序、收缩轴与正确性门槛时才进入实现和 3×8 NPU A/B。
- 当前下一步：在远端仓库外诊断目录复制既有成功 normal-step profiler harness，仅开启 rank0 `record_shapes=true`，以 8 rank/8 NPU 采集一个非周期正常步；原位生成异常报告与 10 节架构报告后，再决定 SOAP 是否存在严格等价候选。
- STEP-068 第一次 record-shapes 采集因 schedule 需要 13 次回调而训练仅 12 步，device kernel dict 为空；训练本身正确。下一步只将 wait 从 11 修正为 10 后独立重试。
- STEP-068 第二次采集有效。SOAP project/project_back 因非 bitwise equal 已拒绝；当前只对 line 299 covariance 做 51 种真实 shape/有效轴 exhaustive bitwise 校验，通过后补齐双报告并决定是否进入单功能实现。
# 当前状态（2026-08-13）

- [x] 完成有效 rank0 record-shapes 8-NPU 正常步采集并原位提取 SOAP shape。
- [x] 完成 SOAP 三类 contraction 初筛及 covariance 51 类 shape/157 轴组合验证。
- [x] 因 covariance 存在 4 个非位级等价真实输入而拒绝候选；未改业务代码、未提交。
- [x] 补齐本次 profile 的异常 JSON/Markdown 和独立 10 节架构报告，并通过正式 schema/章节/状态核验。
- [ ] 基于报告只读聚合未审查源码栈，选择下一项等价边界明确、收益高于噪声的候选。
- [x] 筛选并拒绝 MapTR target mask 与二维点 normalize 向量化：前者机制变慢，后者 3×8-NPU 端到端无收益；均未提交。
- [ ] 继续从高成本纯冗余/数据搬运路径筛选下一项候选。
- [x] 筛选并拒绝 SOAP 二阶矩 `addcmul_` 融合：机制测试通过但 3×8-NPU 端到端性能退化，已回退。
- [ ] 从真实连续 device bubble 与非 profiler 计时上限筛选下一项候选，避免依赖 shape-profiler 聚合时间。
- STEP-073 MapTR target 正负样本 mask 复用已完成 3×8 NPU，但 pooled 普通步仅改善约 1.28%，低于噪声，已回退且不提交。继续筛选其他严格等价、高于噪声的单功能候选。
- 当前下一步：完成 STEP-075 的有效配置和正常步 trace 行级核验；若把每层 `data_valid` 一次性转换为 Python 等值列表的独立同步收益上限明显超过约 22.7 ms，则先做机制级位级等价验证，再进入单一“MapTR 有效数据标量同步移除”候选。性能或等价证据不足即筛掉。
- STEP-075 已完成并拒绝：target-only Python 副本机制位级一致，但三轮 8 NPU 普通步 pooled 退化约 2.60%，必须回退且不提交。下一步回到正常步纯 kernel 与数据搬运的只读筛选；不再依据 `item/local_scalar` host 聚合或独立同步微计时直接推断端到端收益。
- 当前下一步（STEP-076）：按正常 Step 12 的最内层项目 frame 聚合 device self 与 kernel duration，并排除已验证/已拒绝功能族。只允许理论设备收益明显超过约 22.7 ms、且不改变 loss/梯度浮点归约的单功能进入机制验证。
- STEP-076 已完成且无可实施候选。当前下一步（STEP-077）：核验 `maptr_decoder.py:133` 高 host self 与正常 step 连续 device idle 的时间重叠；若只是外层 stack 归因或可回收部分低于约 22.7 ms，则记录筛掉。
- STEP-077 已筛除：line 133 是完整 decoder layer 外层归因。当前下一步（STEP-078）：审计 Hungarian cost D2H 与 SciPy solver；仅当前固定版本存在成熟且能证明 assignment 完全一致的设备端实现时才进入机制验证。
- STEP-078 已筛除：固定版本无设备 Hungarian，完整边界上限仍低于噪声。当前下一步（STEP-079）：只读审计 `maptr_decoder.py:812`，确认是否存在同 step 不变 tensor 的严格等价复用机会。
- STEP-079 已筛除：line 812 为真实动态 output projection。当前下一步（STEP-080）：量化 line 148 的 `zeros_like` 死分配；收益上限不足噪声即不实施。
- STEP-080 已筛除：明确死分配收益仅约 1.434 ms。当前下一步（STEP-081）：按 profile 聚合全项目高频初始化，只保留能证明分配值在读取前被完整覆盖、且同功能上限超过约 22.7 ms 的候选。
- STEP-081 已完成且无候选。当前下一步（STEP-082）：按项目 frame 审计 `to/_to_copy/copy_/cast/TransData`，仅同一 tensor 已满足目标属性且转换可证明无语义/format作用时才允许删除。
- STEP-082 已定位 grad clip 同设备 `.to` 候选。当前下一步（STEP-083）：保持官方 clip-grad 全部算子顺序，只验证条件跳过 `norm.to(first_device)` 是否位级一致且真实 gradients 全在同一 NPU；通过后才考虑最小实现。
- STEP-083 已筛除：位级一致但非 profiler 收益约 1 ms。当前下一步（STEP-084）：审计 contiguous/clone/format 转换；需先证明真实 materialization 可删除且 profiler-off 单功能上限超过约 22.7 ms。
- STEP-084 已完成且无候选。当前下一步（STEP-085）：区分 16 次 `normalize_2d_pts` 调用来源，判断 loss_single 与 GeometricLoss 是否对同一 target 重复计算，并估算整调用删除收益。
- STEP-085 已筛除：仅 1 次可复用 normalize、上限约 14.24 ms。当前下一步（STEP-086）：按行聚合 `loss_single` 重复 `isfinite/all`，只允许同一 tensor/value/shape 的布尔 mask复用。
- STEP-086 已筛除：仅最后层约 3.67 ms 可复用 mask。当前下一步（STEP-087）：量化 loss 默认零 tensor 在启用分支中被立即覆盖的死初始化，收益不足则不实施。
- STEP-087 已筛除：死零初始化上限约 1.624 ms。当前下一步（STEP-088）：提取 target normalize 真实 GT shapes，验证跨 4 decoder 层预计算复用的位级一致性与 profiler-off收益上限。
- STEP-088 已筛除：跨层 normalize 仅约 1.22 ms收益。当前下一步（STEP-089）：提取 SOAP QR 真实矩阵 shape/频次，验证相同 shape batched QR 与逐矩阵 QR 的 Q/R及后续投影是否逐项位级一致。
- STEP-089 已拒绝：完整 batched QR合成周期反而退化。当前下一步（STEP-090）：检查 `geqrf + householder_product/orgqr` 等固定版本官方 Q-only组合；Q或后续投影任一非位级即拒绝。
- STEP-090 已拒绝：非位级且CPU fallback。当前下一步（STEP-091）：只审计 QR前排序/index_select的静态恒等情况，先量化1×1和运行时恒等判断上限，不能缓存旧排序。
### STEP-091 结论：SOAP QR 前排序旁路淘汰

- 状态：已完成。
- 结论：1×1 静态旁路逐位等价，但 profiler-off 摊销仅 1.1648 ms/步，远低于噪声门槛；不修改、不训练、不提交。
- 后续：继续审计 SOAP 周期更新中可静态证明等价、且预估收益明显高于 22.7 ms/步的路径。
### STEP-092：SOAP 大矩阵排序重排等价实现

- 状态：进行中。
- 约束：保留稳定降序索引、状态重排顺序和原始 QR；仅比较逐位等价的 NPU 原生索引实现。
- 门槛：先以 2560/5120 实际频次测 profiler-off 上限，低于正常步噪声门槛则不修改、不训练、不提交。
### STEP-092 结论：大矩阵重排无需替换

- 状态：已完成。
- 结论：当前 `index_select` 的 profiler-off 性能最好，替代实现均退化；不修改、不训练、不提交。

### STEP-093：SOAP QR CPU fallback 与原生等价入口审计

- 状态：进行中。
- 目标：确认 `torch.linalg.qr` 的 host-only 热点是否为 CPU fallback，并仅在现有软件版本中筛选逐位等价、NPU 原生支持的入口。
- 约束：不升级/降级/安装远端框架组件，不使用近似分解，不接受改变 Q 或训练轨迹的实现。
### STEP-093 结论：QR 已为 NPU 原生实现

- 状态：已完成。
- 结论：`torch.linalg.qr` 调用 `aclnnLinalgQr`，不是 CPU fallback；弃用 API 替换没有可接受收益，不修改、不训练、不提交。
- 后续：只筛选不改变稳定排序、Q/R 数值和训练轨迹的 QR 调用编排/数据准备候选。
### STEP-094：SOAP 特征值估算与 power iteration 乘法复用

- 状态：进行中。
- 候选：复用 `m @ o`，或用逐行点积替代只取对角线的第二次矩阵乘法。
- 硬门槛：稳定排序索引、重排后的 power、Q 和状态必须逐位一致；先小形状早停，再测大矩阵真实收益。
### STEP-094 结论：通用矩阵乘法复用淘汰

- 状态：已完成。
- 结论：随机正交基下特征值或 power 在真实代表形状出现非位级差异，禁止用于周期更新。

### STEP-095：SOAP identity 初始基矩阵乘法旁路

- 状态：进行中。
- 范围：仅 `state['Q'] is None` 后刚创建单位基的首次 QR；后续周期更新保持原实现。
- 门槛：24 种真实 shape 的估值、排序、power、Q 全部逐位一致，并证明一次性启动收益值得增加专用分支。
### STEP-095 结论：identity 初始化旁路不实施

- 状态：已完成。
- 结论：24 种 shape 全部逐位一致，但仅一次性节省约 85.6 ms/rank，无法改善稳态周期热点。

### STEP-096：SOAP 独立维度 QR 的 NPU stream 并发

- 状态：进行中。
- 门槛：每个 QR 的 Q/R 逐位一致，且双 2560 或真实配对 workload 的 wall time 有明显改善；否则不重构业务循环。
### STEP-096 结论：SOAP 两流 QR 正式性能失败

- 状态：已完成并拒绝。
- 正确性：函数级逐位一致；8-NPU loss/grad 全有限且与基线分布高度重叠。
- 性能：周期步 13.4→21.843 s（约 +63.0%），普通步约 +1.56%；不运行后续两轮。
- 状态：候选已回退，HEAD `5a37d043`、Git clean、无训练进程、无 commit/push。
- 后续：禁止继续 multi-stream QR；筛选新的严格冗余/数据准备候选。
### STEP-097：SOAP 周期 QR 外严格冗余重新归因

- 状态：进行中。
- 方法：修正 stack 内外层方向，只聚合最内层 SOAP 行；任何候选仍需 profiler-off上限和逐位 state证据。
### STEP-097 结论：QR 外只剩输出缓冲边界值得机制测试

- 状态：已完成归因。
- 结论：排序/重排/转换均低上限；三次 matmul无法逐位代数复用。转入 STEP-098 QR输出缓冲复用。

### STEP-098：SOAP QR Q/R 输出缓冲复用

- 状态：进行中。
- 门槛：现有 NPU原生 `out=` 支持、逐位一致、无别名/同步风险，小中形状先证明 profiler-off收益。
### STEP-098 结论：QR输出缓冲复用淘汰

- 状态：已完成。
- 结论：逐位一致但无 profiler-off收益，并触发内部格式回退警告；不实施。
- 后续：从普通步或跨函数严格重复数据准备筛选下一候选，避免继续消耗在当前QR API包装层。
### STEP-099：SOAP 同 state 等尺寸大 QR 配对审计

- 状态：进行中。
- 范围：远端原位读取已有 checkpoint 元数据，只统计optimizer state的Q/GG shape组合；不拉取产物。
- 门槛：真实存在等尺寸大矩阵且理论周期收益显著越过门槛，才允许机制验证窄分支。
### STEP-099 结论：存在双256/双512窄配对，不存在双大矩阵

- 状态：已完成。
- 结论：52个state含双256、8个state含双512；2560均与空维配对。通用stream失败不能直接否定只并发等尺寸双中矩阵。

### STEP-100：SOAP 仅等尺寸双矩阵 QR 并发

- 状态：进行中。
- 门槛：按52×双256+8×双512真实频次，周期节省需超过227 ms且输出逐位一致；所有非匹配case保持HEAD默认流。
### STEP-100 结论：等尺寸双QR并发亦失败

- 状态：已完成并拒绝。
- 机制：单卡上限与逐位门槛通过。
- 正式结果：Iter11 17.809 s、Iter12 6.429 s，明显差于基线；一轮早停。
- 回退：业务文件与运行时tracked文件均恢复HEAD，Git clean、无训练进程、无commit/push。
- 后续：禁止所有SOAP QR multi-stream方案；返回普通步严格冗余和同步边界。
### STEP-101：denormalize clone/slice回写消除

- 状态：进行中。
- 门槛：真实shape下输出与梯度逐位一致，完整20次profiler-off收益超过22.7ms/step；否则改码前淘汰。
### STEP-101 结论：denormalize回写消除淘汰

- 状态：已完成。
- 结论：逐位/梯度/布局一致，但真实收益仅0.868ms/step，不实施。

### STEP-102：MapTR target类别mask重复构造审计

- 状态：进行中。
- 目标：检查`_get_target_single`多类tag的重复eq/OR是否能一次严格分类复用；先排除STEP-073已验证范围与ignore语义变化。
### STEP-102 结论：ignore mask查找表淘汰

- 状态：已完成。
- 结论：完整值域等价但仅节省1.892ms/step，不实施。

### STEP-103：SOAP投影梯度原地平方复用

- 状态：进行中。
- 门槛：真实559参数shape/state逐位一致，profiler-off收益超过22.7ms/step；保留原第二矩更新顺序。
### STEP-103 结论：原地square复用淘汰

- 状态：已完成。
- 结论：非空Q路径逐位安全，但完整真实shape仅0.514ms/step，不实施。

### STEP-104：SOAP Python scalar overload审计

- 状态：进行中。
- 门槛：固定版本API必须支持0-D tensor且更新逐位一致；完整559 state profiler-off收益超过22.7ms/step，否则淘汰。
### STEP-104 结论：SOAP Tensor scalar overload淘汰

- 状态：已完成。
- 结论：逐位一致但559次dispatcher慢14.2ms，不实施。

### STEP-105：MapTR正负样本计数/data_valid标量链审计

- 状态：已完成并淘汰。
- 目标：确认numel计数可否保持Python整数，并只在最终avg_factor边界转换，减少4层×多类标量同步；不得重复STEP-075的target控制改动。
- 结论：只缓存0-D bool可严格保真，但仅节省0.535ms/step，不实施。

### STEP-106：普通步未闭环热点重新筛选

- 状态：已完成。
- 结论：normal stack最大未闭环业务边界为SOAP project/project_back的tensordot元数据展开。

### STEP-107：SOAP project/project_back第0维收缩表达式

- 状态：已完成并淘汰。
- 结论：`movedim + matmul`真实全频次收益19.096ms低于门槛，且18种4D组合不逐位，不实施。

### STEP-108：SOAP update_preconditioner临时量生命周期审计

- 状态：已完成并淘汰。
- 目标：仅寻找不改变tensordot算术表达、归约维度或lerp顺序的死亡临时量/输出复用边界；真实收益需超过22.7ms且state逐位一致。
- 结论：out buffer逐位一致但真实频次慢8.905ms、常驻272.3MiB/rank，不实施。

### STEP-109：梯度裁剪foreach批量路径

- 状态：已完成并关闭。
- 目标：在当前PyTorch/torch_npu固定版本内验证官方foreach路径是否保持总范数和全部梯度逐位一致，并带来超过22.7ms/step收益。
- 结论：默认None已经与True同速且略快，显式True无收益；不实施。

### STEP-110：SOAP nan_to_num批量预处理

- 状态：已完成并关闭。
- 结论：当前固定torch版本无官方foreach nan_to_num API；保留安全清理，不实施。

### STEP-111：候选收尾与权威状态核验

- 状态：已完成。
- 目标：确认远端HEAD clean、训练/NPU进程为0、无误提交，并校验持久文档已记录STEP-105—110。
- 结论：远端HEAD保持`5a37d043...`且clean，正确容器无训练/NPU进程，本轮无commit/push，记录完整。

### STEP-112：SOAP一阶矩foreach lerp批量更新

- 状态：已完成并淘汰。
- 目标：验证只批量化559个独立exp_avg lerp能否保持完整state/参数逐位并获得超过22.7ms/step收益。
- 结论：修正为保持算术表达的双foreach后逐位一致，但仅节省6.857ms/step，不实施。

### STEP-113：SOAP covariance安全shape窄分支重审

- 状态：已完成并淘汰。
- 目标：排除4个已知非逐位4D axis-1组合，只对其余真实shape/axis评估严格等价改写的全step收益；不改变失败组合或任何其他SOAP路径。
- 结论：多seed与完整state逐位通过，但真实收益仅6.292ms/step，不实施。

### STEP-114：SOAP denominator foreach sqrt/add

- 状态：已完成并淘汰单项。
- 目标：保持`sqrt().add_(eps)`两步数值表达，验证559个真实shape批量路径的逐位与超过22.7ms/step收益。
- 结论：逐位一致但仅节省6.520ms/step，不单独实施。

### STEP-115：SOAP多阶段foreach调度收益上限

- 状态：已完成，暂不实施。
- 目标：逐项验证除法、参数add、weight decay的foreach结果与收益，评估完整多阶段调度功能是否值得结构重构。
- 结论：五阶段独立乐观收益合计19.382ms，仍低于门槛；仅剩二阶矩阶段决定完整方向是否继续。

### STEP-116：SOAP二阶矩foreach与峰值内存

- 状态：已完成机制门禁，转分块验证。
- 目标：保持`mul→out-of-place square→add`表达，验证逐位/收益并量化批量化必须持有的projected gradient与临时量内存。
- 结论：逐位一致、节省10.960ms，但全量额外活跃张量约589.6MiB，不允许直接实现。

### STEP-117：SOAP分块多阶段foreach完整骨架

- 状态：已完成机制门禁。
- 目标：以8M元素预算控制临时内存，在包含列表/分块调度的完整六阶段骨架中验证逐位和超过22.7ms的实际收益。
- 结论：完整结果逐位一致并节省35.315ms，最大临时projected+square约100MiB，允许最小实现。

### STEP-118：SOAP分块多阶段foreach最小实现

- 状态：进行中。
- 目标：单文件实现并通过完整optimizer多stepparameter/Q/GG/一二阶矩逐位门禁；通过后才允许正式8-NPU A/B。
## 当前恢复点（2026-08-13 09:10）

- SOAP 分块 Foreach 候选已通过三轮正式 8-NPU A/B 与功能一致性门禁。
- 下一动作：最终静态检查并提交单一功能 commit；提交后从计划中的下一个未完成性能热点继续，不推送。
- SOAP 分块 Foreach 已提交为 `14d4f23`；下一个优化功能必须从该 clean HEAD 独立开始。
### STEP-120：下一独立性能热点

- 状态：进行中。
- 当前下一步：核验远端 `14d4f23` clean HEAD 和 NPU 空闲状态；读取既有正常步异常报告/聚合摘要，排除 STEP-072—119 已审查功能族后选择一个理论收益明显超过 22.7ms、数值语义可严格保持的候选。
- 提交规则：候选通过机制逐位、3×8-NPU 性能与 loss/grad 门禁后，才以一个功能一个本地 commit 提交；失败候选回退且不提交。
- STEP-120 当前下一步已调整：旧 normal profile 的 SOAP 主热点被 `14d4f23` 改变，先复用既有成功 harness 对新 HEAD 采集 rank0 单 active normal step profile，并产出新的异常 JSON 与独立架构报告；再从新排序选择候选。
### STEP-121：SOAP分块预算收尾

- 状态：进行中。
- 当前下一步：基于真实559 shape比较8M/16M/32M分块的chunk数量、最大临时元素、完整六阶段profiler-off wall和逐位结果。只因同一功能需要收尾而允许amend `14d4f23`；不得新建第二个SOAP foreach commit。
- STEP-121结论：16M/32M只比8M再省0.400/0.548ms，临时峰值增至120.4/220.6MiB；不修改、不amend。当前下一步转STEP-122，归因49.2255ms `NanToNum→Arange` gap。
- STEP-122结论：49.2255ms gap是已完成三轮且仅+0.8897%的冻结骨干模式切换候选，直接关闭。当前下一步为STEP-123细分51.501ms梯度裁剪host边界，只查历史未覆盖的Python参数收集/过滤。
- STEP-123结论：51.501ms gap是已测仅1.06ms收益的同设备`norm.to`候选，没有新边界。当前下一步STEP-124归因16.3845ms Copy→Zero gap并检查严格死写。
- STEP-124结论：16.3845ms Copy→Zero边界仍由已失败的全模型模式切换覆盖，无死写证据。当前STEP-125批量归类剩余top gap并扣除已关闭功能族。
- STEP-125发现`aten::index` 8窗合计53.425ms，是唯一新同功能上限越门槛项。当前STEP-126按项目栈/源码语义拆分，避免把不同动态索引错误合并。
- STEP-126拆分后确定新窄边界：`_get_target_single:1032`三次close-range布尔稀疏权重写。当前STEP-127先核验配置/索引唯一性，并做真实N逐位与完整频次profiler-off门禁。
## STEP-128：loss parse gap 归因（当前步骤）

- 目标：定位提交后 normal profile 中两段合计约 20.909 ms 的 loss parse host gap，确认它是训练必需的 loss 计算、分布式归约，还是可等价收窄的重复日志/字典解析。
- 约束：先只读分析绝对时间窗口、host stack 和源码；不删除 loss、不延迟影响训练控制流的有限值检查、不改变归约顺序或日志数值。
- 门槛：必须形成单一功能边界、严格等价方案，且 profiler-off 乐观上限明显超过 22.7 ms/step，才进入机制实现；否则直接关闭且不创建 commit。
## STEP-129：提交后 host hotspot 去重筛选（当前步骤）

- 目标：以 `14d4f23` 后 normal profile 的 host hotspot 与源码栈为依据，扣除所有已提交或有 profiler-off/正式 A/B 反证的功能族，寻找尚未覆盖且严格等价上限超过 22.7 ms/step 的候选。
- 方法：先做只读聚合，按项目源码最窄栈分组，并分别报告 host self/total、调用次数和真实 device 计算；不把父函数 inclusive time、相邻 kernel gap 或不同源码功能相加。
- 提交边界：只有通过机制位级门槛和三轮 8-NPU 正式验证的单一功能才创建一个 `【npu性能优化】<具体对象/动作>` commit；失败候选不留源码 diff、不提交。
## 待优化项优先级（2026-08-13，按可信耗时/收益上限排序）

> 排序原则：优先采用 profiler-off、device duration 或完整功能 wall；with-stack host self 仅作线索，不直接当作可兑现收益。未知耗时项先测量再插入正式排名。任何候选仍需满足最终功能、loss、匹配结果和优化器状态不发生不可接受偏离。

| 优先级 | 待处理功能 | 当前可信耗时/上限 | 状态与下一动作 |
|---|---|---:|---|
| P0 | SOAP 剩余 2560 维及小矩阵周期 QR | 周期 profile 中 `aclnnLinalgQr` 543 次、device total 22.674 秒；完整周期计算约 23.033 秒 | 最大热点。研究严格保真的分块/降频/AI Core 正交化；任何改变 basis 或更新频率的方案必须先做状态、loss 和长期收敛门禁。固定环境不允许升级依赖。 |
| P1 | MSDA CPU `grid_sample` fallback 现状核验 | CPU fallback 的正式配置频次与 wall 尚未独立量化；新 normal profile 的 NPU MSDA backward device duration 57.884 ms，不能替代 CPU fallback 数据 | 先按实际调用栈区分官方 NPU扩展和 `multi_scale_deformable_attn_pytorch` CPU fallback，原位统计次数、D2H/H2D 和 wall；只有当前 NPU backward 能满足确定性/结果门禁时才允许讨论移除 fallback。 |
| P2 | Hungarian assignment 的 CPU/SciPy 求解 | 已审计完整搬运+求解乐观上限约 22.61 ms/step | 当前无成熟严格等价 NPU solver；继续寻找固定环境可用的精确算法。禁止用贪心/近似匹配替代，避免改变正负样本和 loss。 |
| P3（已关闭） | BEV backbone `pcdet_base_bev_backbone.py:120` | 最新 profile 4步行级拆解：唯一叶子 kernel device self 75.483 ms，即 18.871 ms/step；inclusive device total 71.867 ms/step 是父子 API 重复计数 | 15组真实 Conv+BN+ReLU 分别约14.034/3.029/1.808 ms/step；无 copy/cast/contiguous/format。`cat` 仅0.482 ms/step。低于约22.7 ms噪声门槛且删除/训练期融合会改变功能，不修改、不训练、不提交。 |
| P4 | ConvNeXt/BEV/FPN 与 MapTR 剩余 copy、cast、contiguous、format conversion | `convnext.py:82` host self 35.667 ms、device total 1.319 ms；其他行级项多为约 1～6 ms device | 按最窄源码栈聚合，同一功能才能合并。优先找重复格式化或死 copy；真实卷积、上采样、loss、assignment 计算不改。 |

### 未纳入性能耗时排名的必要验收

- 对 `14d4f23` SOAP 分块 Foreach 做更长周期 8-NPU 收敛验证：长期 loss、grad_norm 周期尖峰、checkpoint 恢复、最终指标与基线对齐。
- 该项是正确性/交付验收，不代表新的可回收耗时；若发现功能或收敛偏离，优先回退或修正现有功能提交。

### 已关闭、默认不重复投入的方向

- SOAP 多流/批量 QR、QR buffer 复用、扩大 Foreach chunk、grad clip 同设备 `.to`、close-range Boolean index 缓存、loss `.item()` 合并、方向余弦/几何 loss 改写、FPN 真实卷积/上采样、L1 `cdist` 手写替代、全模型 train/eval 局部切换。
- 只有输入形状、正式配置、后端实现或新的严格等价证据发生实质变化时，才允许重新打开。

## STEP-130：当前 HEAD 算子性能采集与实测排序（当前阶段）

> 纠正：当前阶段不是直接修改 P0/P1 算子，而是先对 `14d4f23` 采集可比较的当前版本性能。上一节 P0～P4 仅为“待测候选池/历史证据”，不得作为实施顺序；正式优化顺序必须由本阶段的新实测结果决定。

1. 核验环境：仅使用 `mapqr-leicheng`、8张昇腾NPU、现有固定版本；记录 HEAD、配置、8 rank、`torch_npu` 与 `npu-smi`。
2. 采集普通步：排除 warmup、周期 QR、checkpoint 和 profiler 首尾污染，测量 step wall、host self/total、device kernel duration、D2H/H2D、AICPU 与空洞。
3. 采集周期步：单独覆盖 SOAP preconditioner/QR 周期，不能与普通步平均；重新确认当前 HEAD 的 QR 次数和 device/AICPU duration。
4. CPU 路径专项：分别测量 MSDA CPU fallback 与 Hungarian SciPy 的实际调用次数、同步搬运和完整 wall；未知值不得参与耗时排名。
5. 形成当前 HEAD 排名：按同一功能聚合，区分真实 device/CPU wall 与 with-stack 放大值；输出可回收上限和置信度。
6. 只对排名第一且上限越过噪声门槛的单一功能做机制微基准；严格等价通过后才改业务源码，随后执行3轮8-NPU正式A/B和loss/grad门禁。

### 当前可复用证据与必须重测项

- 可复用：`14d4f23` 后 normal-step profile 已存在，可用于候选定位；但 profiler wall 受 with-stack 放大，仍需 profiler-off 实测兑现。
- 必须重测：当前 HEAD 的独立周期步；MSDA CPU fallback 的实际频次/wall；Hungarian CPU完整 wall；BEV line120 的源码子功能耗时。
- 在上述测量完成前，不修改 SOAP QR、MSDA、Hungarian、BEV backbone 或其他业务算子，不创建性能 commit。

## 每阶段性能量化与汇报硬门禁

每个性能阶段（采集、机制测试、正式 A/B、提交）结束后，必须向用户给出可复核的量化结果。没有以下数据，不得宣称“优化完成”或创建性能 commit。

### 统一测量口径

- 代码基准：明确 Before commit、After commit/候选 diff、分支和工作树状态。
- 环境：完整容器名 `mapqr-leicheng`、8张昇腾NPU、固定 config、全局 batch、步数、warmup/排除规则和运行轮数。
- 普通步：至少报告中位数、P95、均值和样本数；说明是否排除首步、SOAP周期步、checkpoint和 profiler 污染步。
- 周期步：单独报告 SOAP 周期步中位数/P95/次数，不与普通步混成一个“平均算子耗时”。
- 端到端：每轮均值、三轮 pooled 均值、吞吐 samples/s、轮间 CV。
- 算子/功能：调用次数、完整功能 wall、device kernel duration、AICPU duration、CPU wall和D2H/H2D；缺失项标记“未采集”，不能用 host self 代替。
- 资源与正确性：最大显存、8 rank/NPU进程核验、loss/grad有限性、loss统计及与基线偏差、错误/fallback/OOM/HCCL状态。

### 统一计算公式

- 绝对节省：`Before耗时 - After耗时`。
- 耗时下降率：`(Before - After) / Before × 100%`。
- 加速比：`Before / After`。
- 吞吐提升率：`(After吞吐 - Before吞吐) / Before吞吐 × 100%`。
- 周期算子摊销：`周期功能节省 / precondition_frequency`，同时保留未摊销周期耗时。

### 每阶段固定汇报表

| 指标 | Before | After | 变化 |
|---|---:|---:|---:|
| 普通步中位数/P95 | 必填 | 必填 | 节省ms、下降率、加速比 |
| SOAP周期步中位数/P95 | 必填或不适用 | 必填或不适用 | 未摊销及每步摊销 |
| 30-step端到端均值 | 必填 | 必填 | 下降率、加速比 |
| 8卡吞吐 | 必填 | 必填 | 提升率 |
| 目标算子/功能耗时与调用次数 | 必填 | 必填 | device/CPU/wall分别比较 |
| 最大显存 | 必填 | 必填 | MiB及比例 |
| loss/grad | 必填 | 必填 | 有限性、统计偏差和结论 |

### 候选失败也必须量化

- 报告逐位/容差结果、Before/After微基准、绝对节省和相对变化。
- 明确失败类型：非等价、收益低于噪声、正式8卡退化、显存代价过高或loss/收敛风险。
- 失败候选恢复到基准 commit，记录Git clean/NPU进程0，不创建commit。

## 永久性能基线与双对照规则

- 永久主基线固定为：`63861dfd920ab9829512b1e4a000eefd1ffcfbea 【loss对齐】随机性移除`。除非用户明确指定新基线，否则后续所有“累计优化效果、极限性能、总加速比、总吞吐提升”都必须与该提交比较。
- 最新已验证 commit 只作为阶段增量基线。例如在 `14d4f23` 上开发下一项功能时，必须同时报告：
  1. `63861df → 候选/新commit` 的累计收益（主结论）；
  2. 直接父提交/当前HEAD → 候选的单项增量收益（辅助结论）。
- 禁止用最新commit替换永久主基线后只报告小幅增量；否则用户无法看到从最初可用版本到当前版本的整体优化效果。
- 若历史基线与当前测试因数据、配置、步数或环境口径不同，必须优先按原夹具在 `63861df` 重测同口径；无法重测时明确标记“历史口径，不可直接比较”，不得混算累计百分比。
- 每阶段固定汇报表增加两列：`相对63861df累计变化`、`相对父提交增量变化`。算子本体新增/消失时，也分别报告基线与当前的调用次数和耗时。

### 已有永久基线数据

- 3轮×8-NPU×30-step：总体 pooled 均值 28.932 秒/step，吞吐 0.2765 sample/s。
- 普通稳态 pooled 中位数 3.186 秒/step，P95 7.074 秒。
- SOAP周期步中位数 271.486 秒，P95 280.307 秒。
- 最大显存 5067 MiB。
- 后续当前HEAD采集必须尽量复用相同配置、全局batch、30-step划分和排除规则，才能计算累计收益。

## STEP-131：`63861df` 独立基线代码与3轮8-NPU性能复测（当前阶段）

- 目标：从永久基线提交 `63861dfd920ab9829512b1e4a000eefd1ffcfbea` 创建独立 detached worktree，在不切换/污染当前 `ascend_npu_optimize@14d4f23` 的前提下，完成3轮×8-NPU×30-step同口径性能测试。
- 隔离：基线 worktree 只用于测试，不修改源码、不创建性能commit、不push；训练产物、日志、checkpoint、资源采样与夹具全部留在远端诊断目录。
- 环境：正式训练只能在完整名称为`mapqr-leicheng`的现有容器，设备0～7；禁止改动驱动、固件、CANN、PyTorch、torch_npu和依赖版本。
- 测试口径：目标config、全局batch8、原训练入口、`MAX_ITERS=30`临时夹具；每轮独立端口/work_dir，启动时核验torch_npu、8个直接rank和npu-smi PID。
- 汇报：三轮分别及pooled报告普通步中位/P95/均值、周期步中位/P95、30-step均值、吞吐、CV、显存、loss/grad范围与错误状态；该结果作为后续永久累计对比基线。
- 当前下一步：只读定位原三轮基线诊断目录/成功夹具和目标config，核验远端当前worktree/NPU空闲，再创建detached worktree。

## STEP-132：按华为官方方法建立“双轨性能优化测试”

- 状态：已完成测试方案设计，待STEP-131基线复测结束后执行。
- 总原则：遵循“明确场景→采集与拆解→定位下发/计算/通信/IO瓶颈→针对性优化→正式验证”；不以HBM空闲率单独判断性能，不在未采集数据前改算子。
- 轨道A（严格等价算子优化）：永久固定`63861df`、8 rank、samples/rank=1、global batch=8、数据/config/步数和排除口径。每个候选先做同输入机制门禁，再做3轮×8-NPU×30-step A/B；同时报告相对`63861df`累计收益和相对父提交增量收益。
- 轨道B（容量/吞吐扩展）：在当前clean HEAD上测试samples/rank=1→2→4，batch=4仍安全且吞吐继续增长时才允许测试8；全程仍为8 rank/8 NPU。该轨道按samples/s、ms/sample、HBM、AICore、通信掩盖和收敛评估，禁止用step time与轨道A直接混算。
- 执行顺序：①完成新机器`63861df`三轮；②同机复测当前HEAD三轮；③采集普通步+SOAP周期步的profiler-off与窄profiler；④形成host/device/communication/IO占比；⑤batch容量单轮筛选；⑥通过档位三轮复验；⑦正式采用前做等样本预算长训、checkpoint恢复和最终指标门禁。
- batch筛选门槛：30/30、8 rank与npu-smi PID一致、loss/grad有限、无OOM/HCCL/fallback；同时记录allocator allocated/reserved与npu-smi HBM峰值。建议HBM总峰值不超过容量约80%作为安全起点；吞吐提升需超过三轮噪声并且P95/CV不恶化。
- 收敛边界：增大batch会改变global batch、每样本优化器更新频率和SOAP触发密度，属于训练策略变化。短跑只证明容量与吞吐；正式提交前必须明确学习率、warmup、训练总样本/epoch、优化器更新次数和评测指标，不得把短跑loss有限等同于收敛对齐。
- 提交规则：算子优化继续一个完整功能一个`【npu性能优化】...` commit；若batch/训练策略最终采用，作为独立、可回退的完整配置功能提交，不与任何算子改动混在同一commit。
- 环境边界：所有训练仍只在`mapqr-leicheng`、8张昇腾NPU和现有固定远端环境执行；不测试16-rank，不改变远端驱动/CANN/PyTorch/torch_npu/依赖版本。本地缺工具可按授权安装并记录。
- 配置权威性前置门禁：本地`custom`中的客户配置候选明确为batch/rank=16、workers/rank=8，与当前性能夹具的batch/rank=1、workers=0不同。在开始batch=1/2/4扩展测试前，必须先只读对比客户配置、`63861df`配置与当前HEAD配置的来源和差异，确认哪个代表客户正式生产口径；未确认前不得把batch=1低显存结论外推到客户正式训练。

## STEP-134：基线与最新代码对齐客户运行配置字段

- 状态：已完成代码修改、配置导入验证、双分支独立提交，以及同机单轮8-NPU×30-step客户batch=16正式A/B。
- 范围：两版均只同步`num_gpus=8`、`batch_size=16`、`workers_per_gpu=8`、`train_loader.prefetch_factor=3`；保留各自数据集清单、路径、SOAP字段、hook和历史优化差异，避免整份替换删除约1474行现有配置。
- 基线代码：从永久算法基点`63861df`派生分支`codex/baseline-customer-runtime-config`，提交`4c37039 【去除随机性固定】客户训练配置字段对齐`。
- 最新代码：当前分支`ascend_npu_optimize`由`14d4f23`前进到`a757f29 【去除随机性固定】客户训练配置字段对齐`。
- 验证：两版Config均成功解析为num_gpus=8、batch/rank=16、global batch=128、num_iters_per_epoch=219、workers=8、pin_memory=True、prefetch=3、SOAP频率10、max_iters=30000；两工作树clean，训练/NPU进程0。
- 对比规则：永久算法起点仍是`63861df`；客户正式配置下的可执行A/B使用`4c37039`对`a757f29`，两边共享同一配置对齐功能。历史batch=1数据保留为旧口径，不与新batch=16直接混算。
- 正式结果：`4c37039→a757f29`的30-step均值37.440→11.745秒（下降68.63%，3.188×），吞吐3.419→10.898 samples/s（+218.76%）；稳定普通23步中位11.501→7.868秒（下降31.59%）；SOAP双步窗口均值279.589→39.227秒（下降85.97%，7.127×）。
- 资源：框架峰值28460→27445MiB/rank，npu-smi HBM峰值45782→44054MiB/65536MiB；两边30/30、exit0、无OOM/HCCL/RuntimeError。
- 正确性边界：loss/grad均有限且范围重叠，没有数量级发散；但这是单轮短跑、两边逐step loss并不相同，不能替代同输入逐位门禁或长期收敛验证。
- 下一步：保留本组作为客户batch=16的单轮累计A/B；若作为正式发布验收，仍需三轮复验和长训收敛/最终指标检查。

## STEP-135：客户batch=16单轮正式A/B（已完成）

- 测试对象：优化前`codex/baseline-customer-runtime-config@4c37039`（父提交永久算法基线`63861df`）对优化后`ascend_npu_optimize@a757f29`（包含刚提交的客户字段对齐代码）。
- 固定口径：完整名称`mapqr-leicheng`，8 Ascend逻辑device、8 rank、batch/rank=16、global batch=128、workers/rank=8、prefetch=3、SOAP频率10、30 step；两边顺序运行，禁止并发污染。
- 普通步口径：排除iter1～2预热、iter11～12与21～22周期窗口、iter30收尾，剩余23步。基线mean/median/P95=11.449/11.501/11.900秒；最新=7.970/7.868/8.443秒。
- 周期口径：iter11+12、iter21+22两组双步窗口。基线281.575/277.602秒，最新39.079/39.375秒。
- E2E口径：30步日志time均值；基线37.440秒、最新11.745秒。对应global batch128吞吐3.419与10.898 samples/s。
- 限制：用户将原三轮计划缩减为单轮，因此本组没有轮间CV/置信区间；历史batch=1的`63861df`数据仅保留旧口径，不与本组百分比混算。

## STEP-136：最新客户负载profile与热点重排（已完成）

- 状态：completed。
- 原因：现有`14d4f23`后profile基于batch/rank=1、workers=2；当前`a757f29`客户配置为batch/rank=16、workers=8，普通步中位已变为7.868秒，周期主步约10秒，旧QR/host-gap排名明显过期。
- 第一步：只读核验HEAD/status/NPU/容器，并枚举远端profile manifest、config hash、step范围、shapes/stack/communication覆盖；已有同口径证据足够则直接复用。
- 重采条件：没有同时覆盖普通步和周期步的`a757f29+batch16`有效profile。重采仍只能在完整名称`mapqr-leicheng`、8 rank/8 NPU、固定客户配置与现有版本执行；原始产物只留远端。
- 采集设计：避免末步checkpoint污染；rank0开启必要的CPU+NPU、record_shapes和with_stack，其他rank只参与真实8卡训练。普通步与SOAP周期可用同一窄schedule覆盖，但必须分别统计。
- 分析输出：正式`anomaly_discovery.json`、异常Markdown、analysis manifest，以及独立`model_architecture_report_<profile>.md`；按多流busy union、通信去重、wait-anchor和AICPU masked ratio规则分析。
- 排名要求：区分完整功能wall、device duration、AICPU、CPU wall、H2D/D2H与with-stack放大值；已关闭方向不因旧kernel名字重复打开。
- 代码门禁：完成最新排名前不修改业务源码、不创建性能commit。排名第一的单一功能仍需严格等价机制门禁、同口径正式A/B及loss/grad验证。
- 采集结果：`a757f29`、batch/rank=16、workers/rank=8、8 rank，ProfilerStep 9～12；14/14、exit0。结构化异常JSON通过正式schema，50个Top bubble有49组完整前后kernel上下文，独立10节架构报告已生成；原始约13.74GB产物只留远端。
- 热点结果：Step10的543次`aclnnLinalgQr`纯设备22.711秒，仍是最大周期热点但属于STEP-093/096/098/100已关闭高风险方向；普通路径`Index`4.433秒/4步、`IndexPut`3.858秒/4步、MSDA backward 3.186秒/4步。调用栈进一步把前三个主要`Index`各约1.501秒/4步归到`geo_loss.py:224/226/228`，无栈的主要`IndexPut`3.931秒/4步高度符合其autograd反向scatter，形成约2.07秒/步的诊断上限。
- 归因边界：profile开启shapes+stack，普通步service/underfeed被显著放大，不能替代profiler-off 30-step；`communication.json`缺失，通信只做kernel/timeline中等置信度判断。

## STEP-137：GeometricLoss有限值索引消除机制门禁（已完成）

- 范围：只研究`geo_loss.py:224/226/228`三处同一功能的有限值筛选及其反向scatter，不夹带SOAP QR、MSDA或Hungarian改动。
- 正确性：覆盖真实shape、全有限/含NaN/空张量，比较loss、每个输入梯度、reduction和异常值语义；默认要求逐位一致，若归约顺序必然改变则必须量化误差并通过训练loss/grad门禁。
- 性能：先做当前固定环境的真实shape profiler-off微基准，目标至少显著削减Index/IndexPut；正式收益必须由8-NPU 30-step父提交A/B给出，profile上限不得当作实际收益。
- 提交：只有正确性和正式A/B均通过才创建一个`【npu性能优化】GeometricLoss有限值索引消除`功能commit；失败则记录实测并保持业务仓库clean。

### STEP-137 后8卡正式复测调整（已完成）

- 用户确认前8张逻辑卡已被占用；本轮正式30-step候选测试改用逻辑设备8～15，仍保持8 rank、同一代码、同一客户配置和同一统计口径。
- 启动前必须归档上一次0-iteration失败启动产生的临时配置、kernel cache等运行副作用，只保留`geo_loss.py`候选diff。
- 启动门禁必须同时证明：完整容器名为`mapqr-leicheng`、`torch_npu`可用、8个直接训练rank、`npu-smi`中的8个唯一训练PID只映射到逻辑设备8～15。
- 当前下一步：清理并归档旧运行副作用，创建独立back8诊断目录，以后8卡启动并核验进程映射。

### STEP-137 完成结论

- 功能：以设备侧masked reduction替代六处有限值Boolean index及其反向IndexPut；`none`/非默认旧参数保留原路径，全有限、部分NaN/Inf、全无效、空张量及完整GeometricLoss前向/梯度的8-NPU同输入门禁逐位一致。
- 正式结果：后8逻辑卡、8 rank、客户batch/rank=16、30/30、exit0、fatal0。相对父提交`a757f29`，严格普通步median 7.868→5.853秒（-25.61%），全30步均值11.745→9.520秒（-18.95%），吞吐10.898→13.445 samples/s（+23.37%）。
- 累计结果：客户配置同口径基线`4c37039`（直接派生自永久算法基线`63861df`）到候选，全30步37.440→9.520秒（-74.57%，3.933×），吞吐3.419→13.445 samples/s（+293.27%）。历史`63861df` batch=1结果不与本次batch=16混算。
- 正确性：30个loss/grad均有限，loss均值296.539、中位281.527、范围208.927～428.746；与父版本单轮分布重叠且无数量级偏离，同输入函数级门禁仍是等价性的主要证据。
- 资源：框架峰值26840MiB/rank，相对父版本27445MiB减少605MiB；本轮未连续采集设备HBM峰值，明确标记未采集。
- 提交：`b36821e 【npu性能优化】GeometricLoss有限值索引消除`，父提交`a757f29`，1文件34增/12删，提交后clean，未push。
- 限制：本轮按用户要求使用后8逻辑卡，而历史对照使用前8逻辑卡；同机同型号但设备子集不同，正式端到端差值带有设备子集混杂。机制微基准1.416～46.162×及普通步约2.10秒节省与profile约2.07秒/step诊断上限相符，支持收益主要来自该功能；发布级置信度仍建议同设备三轮复验。

## STEP-138：MSDA执行分支与backward候选审计（已完成，淘汰）

- 父提交：`b36821e`；永久算法基线仍为`63861df`，客户batch16同口径基线仍为其配置派生提交`4c37039`。
- 资源：所有机制测试、短跑和正式A/B继续只用`mapqr-leicheng`及后8逻辑卡8～15；前8卡不触碰。
- 证据起点：最新batch16 profile中自定义MSDA backward约3.186秒/4步，即约0.796秒/step；这是纯device duration，不等于完整功能wall。
- 第一门禁：当前代码同时存在项目自定义`MultiScaleDeformableAttnFunction_fp32`和MMCV通用fallback，必须先确认有效config各调用点实际走NPU扩展还是CPU grid_sample，不能把两条路径混算。
- 正确性边界：历史CPU fallback用于规避NPU grid_sample随机性；任何候选必须对输出、value/sampling_locations/attention_weights梯度以及重复运行稳定性做同输入门禁，不接受为性能恢复非确定行为或造成loss大偏离。
- 当前下一步：读取MMCV通用实现与`70576d3`差异，结合transfer_to_npu补丁和有效配置确定真实调用分支；随后采集真实调用次数、shape及完整wall。

### STEP-138 完成结论

- 真实训练经`transfer_to_npu`进入项目自定义NPU扩展；历史CPU reference fallback不在本轮激活路径。
- 官方`aclnnMultiScaleDeformableAttentionGrad`主体约3211.402ms/4步，即约802.850ms/step；它是必要数学计算，不作为可删除开销。
- 三个`zeros_like`清零kernel合计仅8.193ms/4步，即约2.048ms/step；顶层zeros_like host total合计约9.154ms/4步，即约2.289ms/step。相对当前普通步约5.853秒只占约0.035%～0.039%，远低于约22.7ms噪声门槛。
- 决策：不改`zeros_like→empty_like`，不做机制微基准、不训练、不提交。即使算子完整覆写buffer，理论收益也不可在正式A/B中可信观测。
- 下一步：转向下一个未闭环独立热点；优先复核BEV backbone line120或聚合后超过门槛的copy/cast/format功能族。

## STEP-139：BEV backbone 行级热点复核（已完成，淘汰）

- 最新客户 profile 共4步；line120 命中960行。其 inclusive device total 为287.468ms/4步，但重复包含 `aten::conv2d -> convolution -> _convolution -> aclnnConvolution`、BatchNorm和ReLU的父子层，不能直接求和。
- 唯一叶子 kernel device self 为75.483ms/4步，即18.871ms/step：卷积56.136ms、BatchNorm 12.116ms、ReLU 7.231ms，分别约14.034/3.029/1.808ms/step。调用数与配置的3个downblock、每块5组Conv+BN+ReLU一致。
- line120未出现copy/cast/contiguous/format；line143的`aclnnCat`仅1.928ms/4步，即0.482ms/step。line127/128/142以及通道注意力也均为真实卷积、反卷积、加法、池化和逐元素计算。
- 决策：可回收边界低于约22.7ms噪声门槛；删除卷积/BN/ReLU或训练期融合BN会改变模型功能。关闭该候选，不做机制测试、不训练、不提交。
- 下一步：按最新profile聚合P4 copy/cast/contiguous/format功能族，只有同一源码功能的严格等价可回收上限越过门槛才进入实现。

## STEP-140：TextLogger跨rank显存统计同步降频（已完成）

- 证据：最新4步profile的copy/cast族按最窄项目栈聚合后，`TextLoggerHook._get_max_memory` 每步一次，line112相关host self约4.694秒/step；源码执行`max_memory_allocated -> NPU tensor -> dist.reduce(MAX) -> item`，存在强同步。with-stack数值只作上限，必须由profiler-off训练验证。
- 功能边界：保持Text/TensorBoard日志每步、loss/grad/lr/step time不变；仅将全局最大显存跨rank同步由每步改为首步及每10步，间隔内复用最近一次累计峰值。峰值统计单调不减，允许诊断字段最多滞后9步；不进入模型计算图、优化器或checkpoint。
- 实现边界：`TextLoggerHook`新增默认值为1的`memory_interval`，保证其他配置行为不变；目标客户config显式设10。两个文件组成一个完整可回退日志功能，只在正式A/B通过后形成一个性能commit。
- 验证：先做构造/缓存频次单元门禁和有效config实例化；再只用`mapqr-leicheng`、后8逻辑卡、8 rank、客户batch16运行30步。相对同为后8卡的父提交`b36821e`报告普通步、周期窗、全30步、吞吐、显存、loss/grad；无收益或异常则完整回退。
- 正式结果：30/30、exit0、Traceback/RuntimeError=0。相对父提交`b36821e`，全30步9.520233→9.313947秒（-2.167%，1.022x），吞吐13.445049→13.742831 samples/s（+2.215%）；普通23步mean/median/P95从5.870696/5.853000/6.121800降至5.715218/5.778270/5.922613秒（-2.648%/-1.277%/-3.254%）；SOAP双步窗口均值35.3495→34.530815秒（-2.316%）。
- 正确性与资源：30个loss/grad全部有限，loss均值/中位/范围311.096343/306.416520/218.127330～442.613650，grad为56.636242/51.480825/45.282260～101.021430；与父版本短跑分布重叠且无数量级偏离。framework显存峰值26851MiB/rank，相对父版本26840MiB仅+11MiB；第30步强制精确刷新。后8卡、8 rank、`npu-smi` PID 8/8门禁通过。
- 累计结果：相对永久算法基线`63861df`的客户同口径派生提交`4c37039`，全30步37.440→9.313947秒（-75.123%，4.020x），吞吐3.419→13.742831 samples/s（+301.955%）；普通步均值-50.081%，SOAP窗口均值-87.649%。历史batch1的`63861df`结果仍不与本次batch16混算。
- 决策：收益超过约22.7ms/step噪声门槛，且功能只影响诊断字段刷新；完成副作用归档和提交前门禁后，按一个完整日志功能创建一个`【npu性能优化】TextLogger显存统计同步降频`commit。
- 提交：`bf9ed6e 【npu性能优化】TextLogger显存统计同步降频`，父提交`b36821e`；2文件13增2删，提交后clean、无训练/NPU占用、未push。
- 下一步：回到最新profile剩余未闭环host-only栈，优先只读审计`lc_fusion.py:311 bev_warp`，扣除异步等待归因后再决定是否有严格等价CPU调度优化。

## STEP-141：LCFusionV2固定网格设备缓存（已完成，淘汰）

- profile边界：有效路径`lc_fusion.py:311`每步一次`self.grid.to(lidar_feat.device)`；固定grid shape为`[1,96,160,2]`，copy host self约133.396ms/step，to/_to_copy仅为父API。device self为0，可能主要是CPU→NPU搬运造成的队列同步归因。
- 实现：只改当前实际命中的`LCFusionV2`；首次设备不同时按原逻辑搬运并把普通属性回写，后续同设备直接repeat。保持普通属性，避免register_buffer进入DDP逐步广播或state_dict；DEPLOY与训练输出公式不变。
- 机制门禁：单后8逻辑NPU上，旧/新完整`grid_sample + conv_lidar`输出、输入梯度、缓存后二次输出均bitwise相同，最大差0；state_dict不含grid。孤立边界0.105088→0.043791ms（节省0.061297ms，2.400x），但不能代表同步上限。
- 下一步：以`bf9ed6e`为父增量基线，仅用`mapqr-leicheng`、后8卡、8 rank、客户batch16跑30步；相对上一轮同设备TextLogger结果量化普通步/全步/SOAP/loss/grad。若无法兑现超过噪声的收益则回退且不提交。
- 正式结果：30/30、exit0、8 rank与`npu-smi` PID 8/8、loss/grad全有限。候选全步9.345220秒、吞吐13.696841 samples/s；普通23步mean/median/P95=5.774374/5.824940/5.999297秒；SOAP均值34.971835秒。
- 相对父提交`bf9ed6e`：全步+0.336%、吞吐-0.335%、普通均值+1.035%、中位+0.808%、P95+1.295%、SOAP+1.277%，无正收益且轻微回归。显存26870MiB，loss/grad有限且无数量级偏离。
- 决策：profile的133ms主要是异步等待归因，真实copy仅0.061ms；候选不具正式可观测收益。已把patch和运行副作用留在远端诊断目录，精确恢复`lc_fusion.py`和`fusion_result.json`；HEAD仍`bf9ed6e`、clean、训练/NPU进程0，不提交。
- 下一步：审计剩余host-only栈`bevformer_encoder.py:602`，同样先拆具体op和源码不变性，禁止从with-stack host self直接推导收益。

## STEP-142：BEVFormer self-attention shape tensor同步锚点复核（已完成，淘汰）

- line602每步创建一个`[1,2]`的`torch.tensor([[bev_h,bev_w]], device=query.device)`，copy host self约74.262ms；紧邻line604创建`[1]`的level-start tensor，copy host self仅0.170ms。
- 两个CPU→NPU小tensor大小同量级，而host归因相差约436x，证明line602主要承接前序异步attention等待，不是74ms真实copy。LCFusion相同模式的正式A/B也已证伪缓存收益。
- 缓存2个整数的纯本体远低于约22.7ms门槛；outer `spatial_shapes`是相机多层shape，不能直接替代self-attention所需BEV `[H,W]`。不改码、不训练、不提交。
- 下一步：当前`bf9ed6e`已包含GeometricLoss和TextLogger两项显著变化，旧`a757f29` profile排名过期；按官方流程在后8卡重采当前HEAD普通+SOAP窗口profile，重新生成异常与架构报告及热点排名后再选候选。
## STEP-144 Result（2026-08-13）

- MapTR冗余`.unique()`候选已完成exact与客户batch16后8卡30-step验证；正确性通过，但相对直接父提交普通步+11.143%、全步+9.660%，判定无增量收益。
- 已精确回退候选，`ascend_npu_optimize`保持`bf9ed6e` clean，无新commit；下一轮继续从当前HEAD profile选择新的独立热点。
## STEP-145：DataContainer页锁定机制验证（当前步骤）

- 证据：客户配置与builder均启用`pin_memory=True`，但MMCV `DataContainer`缺少PyTorch pin-memory协议方法；profile中scatter H2D copy为1828次/4步，host self约18.05秒。
- 边界：候选只允许让DataContainer内部CPU tensor进入页锁定内存；tensor值、shape、dtype、容器元数据、batch顺序、H2D目标设备、stream同步和模型/loss均不得改变。
- 门禁：先用诊断模块证明原始/候选真实batch pinned比例和scatter wall差异，并做逐位语义测试；机制收益超过噪声后才修改单一业务文件并进行8-rank短训、30-step A/B。若pinning成本抵消H2D收益或loss/grad异常，立即关闭且不提交。
- 2026-08-14进展：单文件生产候选已通过协议单测、仓库源码导入断言和后8卡8-rank门禁；每rank非空tensor 3/3 pinned，值/shape/dtype/stride exact，cpu-only不变。真实客户batch16的8-rank 1-step也已exit0，显存24,847MiB/rank，loss/grad有限；原始日志和运行副作用已清理。
- 当前下一步：执行一轮30-step profiler-off正式测试；相对直接父提交`f922c38`判断增量收益，并相对永久基线`63861df`的客户同口径派生结果报告累计收益。未越过噪声或正确性异常则回退不提交。
- 30-step结论：候选普通23步均值5.625957→5.082913秒（-9.652%），但全步9.269933→9.592033秒（+3.475%）、吞吐-3.358%、SOAP+6.698%；判定`REJECT_NO_COMMIT`。相对永久基线客户派生`4c37039`累计仍为全步-74.380%（3.9032x）、吞吐+290.323%。候选已回退，HEAD `f922c38` clean，原始日志/profile/kernel_meta已删除。
- 当前下一步：DataContainer候选关闭。依据当前HEAD留存的脱敏profile报告筛选下一个可独立验证热点；证据不足时只采集最小profile，分析后按长期规则立即删除raw数据。
## STEP-146：DrivingSDK计划R1（当前步骤）

- R0已闭环：MapTR冗余Unique正确性通过但30-step相对直接父提交回归，已拒绝并恢复`bf9ed6e` clean。
- 根据用户最新顺序，先执行DrivingSDK计划；当前选择P1/R1，审计真实活跃MSDA实现与`mx_driving.multi_scale_deformable_attn`的forward/backward等价性和性能。
- DataContainer页锁定候选只保留已验证机制证据，延期到DrivingSDK队列完成之后，不进入业务修改或训练。
- 已完成只读可达性审计：客户固定`mx_driving=1.0.0+gitde13346`提供五输入fp32/ND融合forward+backward，当前项目真实训练走自定义MSDA扩展而非CPU fallback；当前profile为6次forward+6次backward/step，主体合计约0.985秒/step。
- profile的C扩展顶层没有记录完整Input Shapes，子op只能证明部分shape族；不得凭静态注释拼接所有五输入。下一门禁先用不改业务源码的诊断启动包装，在后8卡8-rank真实1-step记录shape/dtype/layout元数据并自然退出。
- 真实合同取得后，构造当前扩展与DrivingSDK的forward及value/location/weight backward比较；覆盖全部shape族、有限性、重复运行稳定性和8-rank一致退出。等价门禁未过不得改业务文件。
- 已取得三类真实合同并完成8-rank数值比较：输出/大部分梯度严格allclose；MapTR/Temporal sampling梯度因近零元素未过严格逐元素阈值，但最大绝对误差≤2.575e-4、NRMSE≤4.101e-7，finite与重复稳定性通过。当前结论仅为近似等价，不授权正式替换。
- 当前下一步：保持业务仓库`bf9ed6e` clean，先运行三类真实shape profiler-off机制微基准。无明显收益则R1关闭；有收益才形成最小候选并以真实1-step loss/grad作为融合归约差异的最终语义门禁。
## 当前恢复点（2026-08-13 23:55）

- DrivingSDK R1 MSDA已完成并提交：`f922c38 【npu性能优化】MSDA切换DrivingSDK融合实现`，父提交`bf9ed6e`，单文件、未push。
- 正式后8卡单轮30-step相对父提交：全步-0.473%、普通均值-1.562%、中位-2.445%、SOAP-3.562%、吞吐+0.475%；P95因SOAP后恢复步抖动+13.355%。相对同配置永久基线`4c37039`累计全步-75.241%、4.0389x。
- 正确性边界：同输入三类真实shape NRMSE≤6.439e-7，30-step loss/grad全有限；不是逐位等价，仍需后续长期收敛验收。
- 资源状态：训练进程0、端口0、工作树clean；远端旧profile已全部按规则删除，顶层profile目录数0。
- 下一阶段：读取DrivingSDK计划中R1之后的未关闭候选，按“只读证据→真实shape机制门禁→1-step容量→单轮30-step正式A/B→单功能commit”推进；DrivingSDK计划阶段完成后，再合并此前通用优化计划。任何新profile在结论归档且不再需要后删除。
## Next Step（STEP-153）

- 只读核验远端`ascend_npu_optimize@f922c38`、工作树、正确容器和NPU状态。
- 静态枚举R2 MapTR target/GT的shift、采样、索引、deepcopy、Python循环和CPU/NPU转换，标出每batch/decoder layer调用关系及可复用不变量。
- 先形成采集清单与理论收益边界；证据不足才设计后8卡8-rank最小profile。未通过真实输入bitwise/equivalent、loss/grad与30-step A/B前不得修改或提交业务代码。

## R2关闭与当前恢复点（2026-08-14）

- `f922c38`活动链已确认客户既有GT shift预计算和4层复用；不重复实现。
- Hungarian索引单次H2D候选8-rank语义exact，但216次真实频次微基准为10.583854→10.786435 ms，0.997935x，拒绝且不提交。
- R2原始profile约3.10 GiB已在报告/合同完整后删除；保留目录只含脱敏分析和夹具，远端顶层profile目录0，业务仓库clean，后8卡空闲。
- 下一步按DrivingSDK计划选择R2之后的下一个活动候选。仍执行“静态可达性→理论上限→机制门禁→必要时1-step/30-step”的顺序；任何新原始profile在结论归档且无复核依赖后立即删除。

## P2当前选择（STEP-155）

- R4：活动但证据不足。仅有通用单次BMM热点，不能归因或证明标准MHA稳定，暂缓且不为它单独立即重采profile。
- R5：当前有效config未实例化包含`mx_driving.bev_pool`的路径，判定不活跃，不测试版本切换。
- R6：四个候选运行时变量当前均未启用。下一步仅做固定torch_npu/CANN安装物的支持证据与进程继承审计，然后只选一个变量规划单变量A/B；不修改业务算子代码。
## 长期门禁：profiling 原始数据生命周期

- 每次 profiling 只在远端原位分析；先保留脱敏指标、异常报告、架构报告、必要输入合同和复现实验夹具。
- 当对应候选已形成结论且后续不再需要回看 kernel/operator/trace 数据时，必须在该阶段收尾时删除原始 profiling 目录，不延迟到以后批量清理。
- 删除前必须核验：报告完整、schema/manifest 通过、无训练或分析进程占用、无进程 cwd 位于目标目录、目标绝对路径严格位于 diagnostics 下且不是符号链接。
- 删除后必须记录目录、释放空间、保留内容和复核结果；训练/A-B 日志、脱敏摘要和必要报告不属于自动删除对象。

## STEP-177：剩余 Conv/TransData 模块级归因（当前阶段）

- 已完成无业务改码的后8卡、8-rank、客户 batch/rank=16、8-step NPU Event 功能块计时；稳定窗口固定为 step3～8，父子模块区间重叠，禁止相加。
- 当前排名（device median）：ResNet 362.8202ms、BevEncoder 311.8138ms、其子 BEVFormerEncoder 229.9559ms、FPN 53.2472ms、MapTR head 47.6234ms、PillarVFE 41.4210ms、SECONDTransFPNV3 31.0124ms、ConvNeXt 26.8460ms。PointPillar scatter 与 BaseBEVBackbone 接近 22.7ms 门槛。
- 下一步优先对 ResNet 与 BEVFormer 做子阶段计时和源码/格式转换审计；只有可独立归因的冗余格式转换、重复计算或无效分支超过 22.7ms 门槛，才形成单功能候选。真实卷积/注意力数学本体不因模块总耗时高而直接替换。
- 本轮逐调用 JSONL、训练日志、work 与 harness 已删除，只保留脱敏 summary/report/SHA；业务 HEAD `f922c38` clean、训练进程0、raw关键文件0。
## STEP-177 point_sampling候选结论（已完成，拒绝）

- 打包BMM通过8-rank逐位与显存/微基准门禁，但两轮新训练进程均重复产生约91秒的iter1/2编译成本。
- 正式复验相对`f922c38`全30步+3.529%、吞吐-3.409%、普通均值+0.284%；SOAP窗口虽-51.368%，仍不满足端到端净收益门禁。
- 候选已回退、无commit；两轮原始训练/checkpoint/harness已删除3,221,813,103 bytes，仅保留脱敏摘要与校验值。
- 当前下一步：保持`f922c38` clean，不重复point_sampling BMM方向；从剩余未闭环热点中选择一个可独立验证、理论上限超过噪声门槛的候选，仍按机制门禁→1-step→单轮30-step→单功能commit推进。
## STEP-178：PointPillarScatter批次向量化（当前候选）

- 现有真实门禁：8rank×8call全部逐位一致，旧→新device 31.946→4.860ms，同步host32.281→5.171ms；真实batch16无空batch/重复坐标。
- 独立功能边界：只把16次逐batchzeros/mask/sum/index-put合并为一次全局flat-index写入；保持SAVE_TENSOR、batch_size、cnt、输出shape/dtype/连续stride和ONNX路径不变。
- 下一门禁：单文件最小patch后做8-rank真实大shape、空batch、重复坐标及输入梯度exact；随后1-step容量与客户batch16×30-step profiler-off。任一关键指标回归即回退，不提交。
## STEP-178-2 门禁边界与清理状态（2026-08-14）

- [x] 修正聚合器的错误通用 `PASS`，改为 `PASS_ACTIVE_VOXEL_CONTRACT`。
- [x] 在脱敏报告中显式保留人工重复坐标失败和适用边界。
- [x] 核验真实 64 次调用无重复坐标，客户 `VoxelGenerator` 源码契约保证坐标唯一。
- [x] 删除本轮原始门禁数据并验证 raw=0。
- [ ] 后 8 卡、8 rank、客户配置 1-step 实训门禁。
- [ ] 通过后运行 30-step profiler-off 测试，并同时对比直接父提交 `f922c38` 与永久基线 `4c37039`。

### STEP-178-3 状态更新

- [x] 后 8 卡、8 rank、客户配置 1-step 实训门禁；1/1 iteration、有限 loss/grad、fatal=0。
- [x] 清理两次无效启动及最终有效轮的全部原始日志/work/checkpoint，raw=0。
- [x] 恢复训练运行时副作用，远端只保留目标单文件候选 diff。
- [ ] 运行30-step profiler-off并按统一口径量化。
- [x] STEP-178 三轮30-step量化完成：`REJECT_NO_COMMIT`。
- [x] 回退PointPillarScatter候选并恢复远端clean HEAD。
- [x] 删除三轮原始work/checkpoint/log及本地一次性脚本，raw=0。
- [ ] 从clean HEAD选择下一个高耗时、可独立回退候选。
- STEP-179（进行中）：PillarVFE最后PFN布局物化消除。客户配置仅一个`num_filters=[32]`最后层；当前`[N,32,32,1]`卷积结果先整体permute+contiguous再max。候选仅把同一逻辑点维max提前到卷积布局，非最后层不变。先做后8卡8-rank真实规模exact/梯度/计时，不通过则不改业务源码。
- STEP-179完成：8-rank真实规模、含tie梯度全部exact，但可回收布局物化仅节省7.906ms/step，低于22.7ms门槛；`CLOSED_BELOW_THRESHOLD`，未改业务代码、未训练、未提交，raw=0。
## STEP-180：剩余 BEV/FPN 活跃模块窄审计（进行中）

- PillarVFE 真实规模 8-rank 门禁已关闭：严格等价但仅节省 7.906410 ms/step，低于 22.7 ms 准入线；未改业务源码、未训练、未提交，原始产物已删除。
- 当前按独立可回收功能审计 `SECONDTransFPNV3`、`ConvNeXt`、`BaseBEVBackbone_FPN`，不得把嵌套模块总耗时相加或把必要卷积/归一化/上采样当作可回收收益。
- `SECONDTransFPNV3` 当前可疑点是四路输出的 `torch.stack(...).sum(...)` 大临时张量；必须先取得真实 shape 与独立 profiler-off 上限，超过 22.7 ms 才允许改业务源码。
- ConvNeXt 活跃配置传入 `normalize=dict(type='BN')`，因此自定义 channels-first LayerNorm 中重复 `(x-u)` 的静态冗余不在当前路径；前向主体为必要 depthwise/pointwise Conv、BN、ReLU 与残差。模块总 device median 仅 26.8460 ms，不把非活跃分支作为候选。
- BaseBEVBackbone_FPN 活跃 forward 由下采样、逐级 lateral/upblock 相加、smooth、upfinal、cat 和客户启用的 channel attention 组成；未发现重复 layout/copy，`cnt += 1` 仅为微小 host 计数。该方向不满足准入条件。
- SECOND 真实配置由 ConvNeXt 的 `[96x160,48x80,24x40,12x20]` 四层和 `[1,2,4,8]` 上采样倍率导出四路同形状 `[batch/rank=16,256,96,160]`。下一步只比较现有 stack+sum 与无 stack 表达的输出/梯度及 profiler-off wall；低于 22.7 ms 即关闭。
- SECOND 门禁结论：pairwise add 在8/8 rank输出与梯度 exact，old forward/full pooled median 10.121765/10.404022 ms，candidate 2.058282/2.454373 ms，节省8.063483/7.949649 ms，低于22.7 ms；`CLOSED_BELOW_THRESHOLD`，无业务改动、无训练、无提交。8个逐rank raw及临时脚本/日志共30,901 bytes已删除，仅保留summary/report/SHA。
- 一次同时读取三个完整源码文件的命令输出超出上下文并被截断；已改为先定位 class/forward 行号、再按单文件窄范围读取，不以截断输出形成结论。
- 容器重启后 `bash -lc` 会在profile启动中触发额外torch探测并使只读exec超时；改用 `bash --noprofile --norc -c` 后容器、torch_npu和8设备检查正常。遗留的本轮只读shell已精确终止，不影响训练或同事前8卡任务。
- 历史 diagnostics 只读复核：`trace_view.json/kernel_details.csv/operator_details.csv/op_summary/communication/step_trace` 原始标志目录为0，`PROF_*`目录为0；当前没有遗留 profiling raw 可删。
## STEP-181：DrivingSDK 队列完成度审计与下一方向选择（进行中）

- 先按当前权威HEAD `f922c38` 修正计划文档中仍写 `bf9ed6e` 的历史交接状态；不得据旧HEAD重复R0或其他已闭环实验。
- 逐项用现有脱敏报告映射 R0/R1/R2/R4/R5/R6/R7/R8/P3 的完成、拒绝、非活跃或证据不足状态；只有尚未闭环且当前活跃的方向才能继续。
- DrivingSDK计划明确把SOAP QR强行替换、block化或升级软件栈列为已否决方向；除非出现新的固定环境等价实现证据，不重复该方向。
- 当前不占NPU。下一步先形成队列状态矩阵并核对 `custom/DrivingSDK优化研究与实施计划.md` 是否与根目录版本一致，再从通用计划选择仍有新证据空间的候选。

| 队列项 | 当前权威状态 | 结论 |
|---|---|---|
| R0 MapTR Unique | 已完成exact与30-step | 正确但端到端回归，拒绝、已回退 |
| R1 DrivingSDK MSDA | 已完成并提交 `f922c38` | 当前唯一DrivingSDK新增commit |
| R2 MapTR target/GT | 已完成静态+窄profile | 客户缓存/前移已生效，剩余项无合格候选 |
| R4 标准MHA | 8-rank实测 | forward仅约1.55ms/step，低于门槛 |
| R5 BEV Pool | config/profile可达性 | 当前路径未实例化，关闭 |
| R6 运行时开关 | 单变量A/B/拓扑审计 | TASK_QUEUE/COMBINED回归，affinity证据不足，expandable不适用，整体关闭 |
| R7 internal format | 8-rank兼容/性能 | 数值可接受但Conv/ConvTranspose均回归，关闭 |
| R8负载均衡 | 8-rank 30-step相关性 | rank离散与样本复杂度无显著相关，关闭 |
| P3 | profile命中审计 | Sparse/Pillar/自定义/HCCL均不满足前置条件，关闭 |

- 根目录与 `custom/` 下DrivingSDK计划SHA256相同。DrivingSDK队列已按计划全部闭环，后续转入通用计划。
- 通用计划中尚未正式闭环且可能影响主要Conv/MatMul计算的固定环境候选是HF32单变量开关；先只读确认当前显式关闭位置、API和真实活跃算子覆盖，再决定是否做后8卡隔离门禁。conv与matmul必须分开，不得组合。
- 远端 `f922c38` clean；`tools/train_spetr.py:412-413` 在NPU初始化前显式关闭Conv与MatMul HF32，两行来自2026-08-04完整工作区提交，不是后续独立性能决策。固定torch_npu 2.7.1暴露独立 `torch.npu.conv.allow_hf32` / `matmul.allow_hf32` 属性。
- 固定版本默认值为Conv HF32=True、MatMul HF32=False；客户入口把Conv从默认True改为False。仓库内DrivingSDK `MapTRv2.patch`还在A3/arch35分支显式建议MatMul HF32=True。先从Conv单变量开始，因为当前最大活跃必要计算是ResNet/Conv且这是恢复固定版本默认；MatMul保持False不动。
- 权威客户配置文件已从既有记录确认：`projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py`，batch/rank16、workers8；当前入口图像scale字段为1333×800，仍需读取实际resize/crop链得到送入ResNet的最终shape，不能直接用配置字段猜微基准。
- 权威配置使用7路相机、ResNet34；源码注释给出四stage输出约`[64,136,240]→[128,68,120]→[256,34,60]→[512,17,30]`，但训练pipeline含`ResizeCropFlipRotImageOnlyResize2D3D`与快速Pad，仍需用data_aug_conf或运行时shape确认最终输入。
- 训练增强合同为src_size576×1024、resize随机范围±5%、crop_h=0、flip=False、rot=0，随后Pad到32倍数；因此ResNet输入空间shape是动态shape族，不应只测单一静态尺寸。需从变换实现枚举可能的pad后尺寸，或以真实batch探针取得频次。
- 更正：变换实现无论随机resize结果如何，最终crop窗口固定为src_size 576×1024；Pad32不再改变尺寸。因此ResNet真实空间shape固定，合并7相机后输入合同为`[112,3,576,1024]`。上一条“动态shape族”已由源码实现证伪，不据此设计多shape测试。
- 首次全目录递归grep因共享盘扫描超过24秒超时，未改变任何状态；已改用`git grep`限定tracked文件，后续不重复宽扫描。
- Conv HF32首次8-rank夹具在算子执行前失败：MMCV BaseModule的`eval()`返回None，链式`ResNet(...).npu().eval()`令model变量为None。已用CPU短进程证明原对象仍callable、`eval()`返回None；修正为移动设备和eval分步调用。该轮无HF32结果，不计入候选结论。
- 修正版Conv HF32门禁8/8、exit0：完整ResNet真实输入False/True pooled median=362.725170/284.700794ms，节省78.024376ms、-21.511%；代表Conv/Deconv分别-22.543%/-6.104%。全部输出/输入梯度/权重梯度finite-mask一致，最大NRMSE=2.787196e-4、max_abs=7.255077e-4，非逐位一致。判定`QUALIFIED_FOR_1STEP`，不等于最终采用。
## STEP-181 Conv HF32 单变量门禁

- [x] 固定版本 API/default 与客户入口覆盖审计（Conv 默认 True、客户显式 False；MatMul 保持 False）。
- [x] 后 8 卡、8 rank、真实 ResNet shape 机制性能与数值门禁。
- [x] 后 8 卡、8 rank、客户配置 1-step 容量/正确性门禁，并清理原始数据。
- [ ] 单轮 30-step profiler-off 正式性能测试，同时对比直接父提交与永久基线。
- [ ] 根据净收益、数值边界和噪声判定提交或回退；完成原始数据清理和操作记录。

### STEP-181 完成状态

- [x] 单轮 30-step profiler-off 正式性能测试，同时对比直接父提交与永久基线。
- [x] 判定 `REJECT_NO_COMMIT` 并回退候选；远端/本地一次性原始产物完成清理。
- [x] Conv HF32 方向关闭，不再重复测试。
## STEP-182：下一高价值候选筛选（已完成）

- [x] 恢复文件化计划并复核昇腾 7.3 官方“先采集拆解、后定向优化”及并行策略边界。
- [x] 复核 DrivingSDK 计划完成矩阵与通用计划所有未关闭项，排除已拒绝/不适用/低于阈值方向。
- [x] 用现有脱敏 profile/module timing 证据计算剩余候选的可回收上限，选择唯一下一候选。
- [x] 若现有证据不足，仅采集最小必要证据；本轮使用正确容器的 CPU-only DataLoader 探针，未占 NPU，结论归档后已删除原始数据。
- [x] 所有入选线索均按机制/数值/必要性门禁闭环；本阶段没有新的合格业务候选，无需1-step或30-step，不创建commit。

### STEP-182 当前唯一线索

- 只读归因当前HEAD normal-step的MapTR loss入口GT列表`.to(device)`（历史host-context bubble003）。
- 先证明调用次数、真实源/目标device、总字节、完整wall和是否存在可严格等价的打包/上游复用边界；上限低于约22.7ms或只是必要搬运即关闭。
- 在证据满足前不修改业务代码、不启动正式训练、不重开已关闭HF32/QR/格式转换方向。

- [x] bubble003 GT列表`.to(device)`：实际self约1.3～2.5ms且为必要搬运，低于门槛，关闭。
- [x] 只读审计bubble004 gradient fingerprint hook、zero_grad和DDP unused-parameter搜索，区分必要训练语义与纯诊断开销。

- [x] fingerprint功能在客户配置中关闭，zero_grad/backward/clip/step均为必要语义；DDP搜索self仅约3.416ms。
- [x] 后8卡8-rank客户配置3-step只读式梯度存在性探针；任一trainable参数无梯度即关闭`find_unused_parameters=False`候选。

- [x] 3-step探针：rank0每步142/701个trainable参数无梯度，关闭unused检测不适用；候选关闭并完成raw清理。
- [x] 汇总当前HEAD剩余热点证据覆盖；现有高位热点均已有提交、拒绝或必要性闭环，暂未发现上限超过22.7ms的新安全候选。

- [x] 当前profile BatchMatMul唯一归到已拒绝的BEVFormer point_sampling；Unique/Nonzero/Index/Reduce均有既有闭环，不重复。
- [x] 对活跃VectorizeLocalMap无条件MAP_SHIFT三行调试输出做CPU-only真实DataLoader频次/墙钟门禁；仅2次触发/6 batch，低于门槛并关闭。

- [x] 正确容器CPU-only取6个客户batch，统计`idx/choice/hash`各2行、总6行；不初始化NPU，后8卡进程始终为0。
- [x] 删除本轮远端5个原始探针文件（12,167 bytes）及本地一次性脚本/pycache；远端`f922c38` clean、profiling raw关键文件0。
- [x] 转入已提交优化的更长正确性/收敛验收方案；启动任何正式训练前继续等待用户确认可用卡时段。

## STEP-183：已提交优化的扩展正确性/收敛验收（已完成）

- [x] 只读核验`f922c38`及直接父提交`bf9ed6e`、客户config、runner/epoch、初始权重、checkpoint interval、resume和评测入口。
- [x] 固定同数据/config、同后8卡、同8 rank、同样本预算的父提交/当前HEAD A/B合同；保持生产自然随机语义，不通过重新固定随机种子制造逐step对齐。
- [x] 明确扩展步数、预计时长、loss/grad轨迹统计、SOAP周期统计、checkpoint保存与恢复、最终指标/容差和早停条件。
- [x] 启动前核验正确容器、`torch_npu`、后8卡健康/空闲、端口、Git clean和canonical harness SHA。
- [x] 顺序执行父提交与当前HEAD，禁止并发污染；每轮结论提取后删除不再需要的原始日志/work/checkpoint/profile，只保留脱敏摘要与校验。
- [x] MSDA长期数值/收敛门禁通过，形成发布级验收结论；验收本身未创建性能commit。

### STEP-183 固定合同

- 训练对象：直接父提交`bf9ed6e`与当前`f922c38`，各自独立只读worktree；客户config SHA256=`217ec2e7...b721`，canonical harness SHA256=`10ad92c...e0fc`。
- 训练口径：`mapqr-leicheng`、后8逻辑卡8～15、8 rank、batch/rank16、global batch128、workers8、prefetch3、自然随机语义、完整4 epoch。数据28130帧，`28130//128=219 iter/epoch`，总876 iter。
- 时长预算：按当前约9.3s/step估算每版约2.3小时，顺序总计约4.6～5小时；禁止两版并发。
- checkpoint：配置interval1000大于876，但MMCV `save_last=True`，要求末步产生latest/iter_876；随后每版从该checkpoint恢复并续跑到881，验证模型、SOAP optimizer、runner与hook状态可恢复。
- 正确性裁决：876+恢复5步均完成，全部loss/grad finite，无OOM/HCCL/RuntimeError；按epoch统计loss/grad mean/median/P05/P95/首末趋势，当前HEAD相对父提交不得出现数量级偏离、持续发散或恢复后不连续。独立自然随机轨迹不做逐step相减。
- 性能/资源：同时记录全步与各epochstep mean/median/P95、SOAP周期、吞吐、framework/npu-smi HBM峰值；性能只作扩展稳定性复核，仍以既有同口径30-step为主结论。
- 评测边界：config训练内evaluation间隔极大，不会自动运行；仓库存在8卡测试入口，但test数据为远端对象存储引用。训练完成后先只读确认结果评测可执行性，再决定是否用两个末步checkpoint顺序评测；不可执行则明确标记“最终任务指标未验证”，不得伪称完成。
- 早停：任一版本loss/grad非有限、连续明显爆炸、OOM/HCCL/fatal、rank缺失或后8卡之外出现本轮PID，立即终止该轮并保留最小失败摘要。

### STEP-183 启动错误记录

- 父提交首次launcher在0 iteration、0 NPU进程时退出；原因是detached worktree缺少主仓库中Git忽略的现成MMCV编译扩展，Python从隔离worktree加载到不完整`mmcv`。不计入训练结果。
- 处理边界：禁止安装/重编远端依赖；先证明两提交未修改相应依赖源码，再仅在隔离worktree按相对路径链接主仓库现有同版本`.so`，随后重做Git/config/harness/rank/NPU门禁。失败raw在脱敏原因/SHA生成后删除。
- 第二次启动同样在0 iteration/NPU进程0时退出：`docker exec -e PYTHONPATH=...`覆盖了容器原有CANN/TBE路径，导致首次NPU初始化`AclSetCompileopt`失败。容器原环境可正常import tbe。第三次改为容器内前置项目路径并保留原`$PYTHONPATH`，不改环境安装。
- 第三次启动通过正式门禁并正在运行：唯一`LOCAL_RANK=0..7`、`WORLD_SIZE=8`，主进程一一映射物理后8卡4/0～7/1；截至29/876，loss/grad全有限，无fatal。
- 300步区间统计首次命令因PowerShell内联转义把正则边界写成字面量，解析字段为None并在statistics.mean早退；未触碰训练。改用已验证here-string模板并显式过滤None后成功，不重复内联转义方案。

### profiling 原始数据固定清理规则（2026-08-14再次确认）

- 每轮 profiling 只在远端独立诊断目录临时保存 raw trace、kernel/operator/op_summary、communication、数据库及导出目录。
- 完成远端原位分析、脱敏统计/报告和 SHA 校验且后续不再复用后，按“解析绝对路径 → 限制在本轮诊断目录 → 拒绝符号链接 → 删除 → 复核文件/目录数为0”执行清理。
- 当前 STEP-183 是 profiler-off 长期收敛验收；仍用于 checkpoint 恢复和可能最终评测的训练日志/work/checkpoint 不属于 profiling raw，不提前删除，待对应验收闭环后按既定合同清理。

### STEP-183 父提交 epoch2 里程碑

- [x] `bf9ed6e` 完成 iter220～438；219条 loss/grad 全有限，严格 fatal=0。
- [x] loss mean/median/P05/P95=`60.518547/60.205100/44.553130/78.473120`，前20步 mean=`74.245135`、末20步 mean=`51.337850`，相对 epoch1 继续下降。
- [x] normal 175步 mean/median/P95=`6.397589/6.294000/7.699600s`；SOAP 44步=`16.943227/17.437500/27.186450s`；全 epoch 吞吐=`15.029918 samples/s`。
- [x] framework memory 峰值保持 `27086 MiB`；进入 epoch3 的前4步仍为 `27086 MiB`，暂无持续增长证据。
- [x] 继续同一任务至 epoch3 边界657，不重启、不并发；届时按相同口径统计。

### STEP-183 父提交 epoch3 里程碑

- [x] `bf9ed6e` 完成 iter439～657；219条loss/grad全有限，严格fatal=0。
- [x] loss mean/median/P05/P95=`43.356168/43.102500/31.938700/55.758720`，前20步mean=`46.933555`、末20步=`40.682230`，较epoch2继续下降。
- [x] normal 175步mean/median/P95=`6.446834/6.352000/7.740600s`；SOAP 44步=`16.845659/17.152500/25.914050s`；吞吐=`14.995146 samples/s`。
- [x] framework memory峰值继续保持`27086MiB`，进入epoch4后未增加。
- [x] 完成最终epoch至876、核验末步checkpoint并执行resume5步。

### STEP-183 父提交最终结果与恢复门禁

- [x] epoch4 iter658～876：loss mean/median=`35.785836/35.168800`，前20步mean`39.885635`、末20步`34.604210`；全部loss/grad finite，fatal0。
- [x] epoch4 `time` mean=`8.604228s`、normal mean/median/P95=`6.470903/6.303000/8.055100s`、窗口mean=`17.089045s`、throughput=`14.876407 samples/s`、`memory` max=`27173MiB`。
- [x] 主训练876/876自然退出；`iter_876.pth`大小1607991785 bytes，`latest.pth -> iter_876.pth`。
- [x] checkpoint恢复至881：8 rank映射后8卡4/0～7/1，checkpoint加载、fatal0，生成`iter_881.pth/latest`。
- [x] MMCV checkpoint保存标记为876但resume meta为iter875，故恢复日志为876～881共6条而非5条；6条loss/grad均finite，属于现有框架编号语义，记录后接受。
- [x] 顺序启动当前`f922c38`相同876步合同，禁止并发父版本。

### STEP-183 当前MSDA版本启动门禁

- [x] `ascend_npu_optimize@f922c3897255`、Git clean、config/harness SHA与父版本一致。
- [x] 唯一`mapqr-leicheng`容器、逻辑设备8～15、8 rank、端口29926；rank0～7映射物理4/0～7/1。
- [x] 当前版本已进入真实iteration；11/876时loss/grad finite、fatal0、`memory=26482MiB`。
- [x] 继续至epoch1边界219，按父版本相同口径统计并优先追加GPU `time/memory/loss/grad_norm`参考。

### STEP-183 当前MSDA版本 epoch1

- [x] iter1～219全部`loss/grad_norm` finite、fatal0；`loss` mean/median=`139.128243/109.554200`，前20步mean`342.197115`、末20步`72.327595`。
- [x] `time` mean=`8.130699s`、normal mean/median/P95=`5.798000/5.639500/7.176000s`、固定窗口mean=`16.458238s`、throughput=`15.742805 samples/s`、`memory` max=`26842MiB`。
- [x] 与父epoch1趋势/数量级一致；当前相对父全epoch `time` -7.113%、normal mean -8.363%、窗口 -2.974%、throughput +7.658%。
- [x] 继续至epoch2边界438，降低SSH轮询频率，避免握手限流。

### STEP-183 当前MSDA版本 epoch2

- [x] iter220～438全部`loss/grad_norm` finite、fatal0；`loss` mean/median=`59.415967/58.483800`，前20步mean`72.269845`、末20步`51.273245`。
- [x] `time` mean=`7.971680s`、normal mean/median/P95=`5.864211/5.718000/7.032600s`、固定窗口mean=`16.353659s`、throughput=`16.056840 samples/s`、`memory` max=`27085MiB`。
- [x] 相对父epoch2：全epoch `time` -6.395%、normal mean -8.337%、窗口 -3.480%、throughput +6.833%、memory -1MiB；loss末段几乎一致。
- [x] 继续至epoch3边界657，保持低频SSH查询。

### STEP-183 当前MSDA版本 epoch3

- [x] iter439～657全部`loss/grad_norm` finite、fatal0；`loss` mean/median=`43.287042/43.381800`，前20步mean`46.199335`、末20步`40.483295`。
- [x] `grad_norm` mean/median=`43.943570/43.387100`；`time` mean=`8.261699s`，normal mean/median/P95=`6.123983/6.000000/7.654600s`，固定窗口mean=`16.763977s`，`throughput (samples/s)=15.493182`，`memory` max=`27085MiB`。
- [x] 相对父epoch3：`time` -3.214%、normal mean -5.008%、固定窗口 -0.485%、`throughput (samples/s)` +3.321%、`memory` -1MiB；`loss`末20步相差约0.489%，收敛趋势一致。
- [x] 继续同一任务至876自然退出，核验epoch4、checkpoint/latest后执行当前版本resume门禁。

### STEP-183 当前MSDA版本最终结果与恢复门禁

- [x] epoch4 iter658～876：`loss` mean/median=`35.841326/35.668500`，前20步mean`39.575625`、末20步`35.016005`；全部`loss/grad_norm` finite、fatal0。
- [x] epoch4 `time` mean=`8.299005s`、normal mean/median/P95=`6.204086/6.042000/7.685200s`、固定窗口mean=`16.631068s`、`throughput (samples/s)=15.423537`、`memory` max=`27175MiB`。
- [x] 全876步 `time` mean/median/P95=`8.165771/6.050000/25.248000s`、`throughput (samples/s)=15.675189`、`memory` max=`27175MiB`；相对父版本分别为`time -5.077%`、`throughput (samples/s) +5.348%`、`memory +2MiB`。
- [x] 主训练自然退出并生成`iter_876.pth/latest`；恢复日志按既有MMCV meta语义记录876～881共6条，全部finite、fatal0，`loss` mean/range=`34.500800/29.4106～38.9806`，生成`iter_881.pth/latest`并自然退出。
- [x] 最终评测可达性预检：当前提交原始config因`lidar_type`与`InternalDatasetTrackStream`签名不兼容而在dataset构建前失败；仅内存移除该旧字段后，远端数据集构建90秒仍未返回。最终任务指标标记为“未验证”，不得以本地历史评测表代替。
- [x] 生成最终脱敏对比摘要；删除不再需要的训练日志、resume checkpoint和夹具，仅保留父/当前两个iter876 checkpoint供后续评测配置修复后使用。
- [x] 裁决：保留`f922c38 【npu性能优化】MSDA切换DrivingSDK融合实现`；长期`loss/grad_norm`、checkpoint恢复和相对父版本性能门禁通过，不为验收另建commit。
- [x] 回到GPU主要参照并完成当前HEAD最小稳态SOAP窗口profile；候选裁决`CLOSED_NO_NEW_FIXED_ENV_EQUIVALENT`，未重开已有证据关闭的SOAP方案。

## STEP-184：当前HEAD稳态SOAP周期窗口重新profiling（已完成）

- [x] 使用`ascend-profiling-anomaly`技能完整读取kernel数据指南、异常rulebook、正式schema、参考实现和10节架构报告模板。
- [x] 固定采集合同：`f922c38`、正确容器、后8逻辑卡、8 rank、客户batch/rank16、rank0 profile、`wait=9/warmup=1/active=2`、12 step、Level0、`record_shapes/with_stack=true`、checkpoint关闭。
- [x] 采集自然完成：12/12、全部`loss/grad_norm` finite、fatal0；8个主rank映射物理4/0～7/1，profile覆盖Step10/11并导出约6.90GB raw。
- [x] 远端原位生成`anomaly_discovery.json`、异常Markdown和独立模型架构Markdown；schema errors=0、架构报告10/10节。
- [x] 周期Step10：`service=62376.371ms`、device busy=`23903.239750ms`；543次`aclnnLinalgQr_QrAiCPU_Qr`=`22641.383956ms`，占busy 94.721%、占Step10-Step11 busy差99.118%，分类`AICPU_EXPOSED_NOT_ALLOWED`。
- [x] shape归因：4次`2560x2560`=`16147.768347ms`，其次22次768=`2188.898305ms`、43次512=`1558.070687ms`、6次1024=`1459.024009ms`、181次256=`1048.221737ms`。
- [x] 候选裁决`CLOSED_NO_NEW_FIXED_ENV_EQUIVALENT`：新结果与历史22.711s仅差约-0.31%；geqrf/orgqr/householder、batch、out-buffer和multi-stream均已有拒绝证据，block/降频会改变SOAP语义，固定环境禁止升级。无改码、无commit。
- [x] 删除本轮profile、trace、operator、work、harness和一次性分析脚本；只保留脱敏双报告、JSON、schema、候选裁决、SHA与清理报告，复核raw0/Git clean/后8卡空闲。
## STEP-185 客户评测数据可达性与 DrivingSDK MSDA 同 checkpoint A/B（2026-08-14）

### Current status

- [x] 定位 NPU 服务器上与客户配置文件名一致的评测镜像；原始 `ann_file/flag_file` 路径不写入本地记录，只保存路径 SHA256。
- [x] CPU-only 构建 `InternalDatasetTrackStream`，确认 `dataset_len=25287`，并成功读取 `sample0` 的图像、点云、标定和 voxel 等完整字段。
- [x] 用同一客户 checkpoint、相同连续样本、`VIS_RATE=1`、后 8 卡和 8 rank，对 `bf9ed6e` 与 `f922c38` 完成 16-sample 及 512-sample 推理 A/B。
- [x] 16-sample 原始结果结构/shape/finite 全一致；512-sample 按固定 shape 与可变长结果分开分析，避免错误地按数组下标比较不同数量的预测。
- [x] 512-sample 日志字段 `F1/Precision/Recall/IoUMean/TP/FP/FN/TN` 完成同口径对照；同时记录 `task/s` 与 `elapsed`。
- [x] 完成远端 raw、子集标注、结果 pkl、日志、临时 config、probe、compat 和父 worktree 清理；只保留脱敏摘要与 SHA。
- [ ] 客户容器缺少 canonical `ortools`，且远端禁止安装；完整 25,287 样本的 canonical 绝对 RG 指标需在客户提供既有兼容评测环境后执行。

**Status:** in_progress

## STEP-216-A-MF：Brockett局部筛选包独立审计修复（2026-08-15）

- [x] 用`step216_source_contract_v2`固定adapter/config/checkpoint/SOAP/community policy以及8个harness源码的name/bytes/SHA256和真实SOAP schema；manifest排除自身。
- [x] candidate复用预计算`power_iter/trace_norm`，删除重复`C@Q`；保留活动23类/543次，5120及未知shape fail-closed回退。
- [x] 删除4列marginal proxy，改为559个真实`exp_avg`经`project_back(oldQ)->project(newQ)`的全轴作用比较，记录逐tensor/global relL2。
- [x] 改为至少3次交替的完整543调用cycle Event/wall配对判定；显存按candidate peak减baseline peak判定，周期门槛仍为`>227ms`。
- [x] ready记录容器/宿主PID，controller严格绑定后8die的8个direct rank host PID；统一1200秒、TERM后5秒KILL，并做端口/进程/NPU postflight归零。
- [x] Python编译、6项policy测试、5项纯stdlib合成测试、source package identity和diff-check通过；本机无Bash/WSL发行版，shell只完成静态人工审计，未声称`bash -n`通过。

**Status:** complete_static_fail_closed_not_npu_executed

## STEP-216-A-RUN：world8核心局部测试唯一执行（2026-08-15）

- [x] 上传固定source package并逐SHA/bytes核验；正确容器内11项测试、两份`bash -n`、HEAD/allocator-only tracked diff、固定config/checkpoint/SOAP、端口/训练/NPU空闲均通过。
- [x] 直接入口因SFTP文件无执行位`rc126`，记`failed_start=1/effective=0`；经主任务授权仅以`bash host_launcher`纠正一次，不改源码/算法/参数。
- [x] 纠正入口的runner因adapter位于`BUSINESS_REPO/diagnostics`而违反“adapter必须在repo外”路径断言，`rc1`且output/ready/rank均未创建；host清理命令自匹配被TERM，最终外层`rc143`掩盖原始rc1。
- [x] 不再重跑；确认world8/NPU Event样本为0、训练/相关进程0、端口空、后8 NPU进程0，并永久保留preflight、outer、Docker event根因摘要、failure summary和manifest。

**Status:** complete_rejected_prelaunch_no_core_sample_no_rerun

## STEP-216-A-RECOVERY：启动合同本地静态修复（2026-08-15）

- [x] 将入口合同改为显式`TOOL_ROOT`，realpath强制tool/adapter/output均位于`BUSINESS_REPO`之外；adapter必须在`TOOL_ROOT/harness`、output必须在`TOOL_ROOT/runs`。
- [x] 删除所有`pkill -f OUTPUT`/字符串进程扫描；runner与host分别记录精确PID/PGID，只按记录的进程组执行TERM→等待→KILL。
- [x] 新增纯stdlib路径边界和cleanup不可自匹配测试；刷新source contract并完成pycompile、6+7项测试、两shell真实`bash -n`和diff-check。
- [x] 仅生成后续可恢复静态包；未上传、未连接远端、未启动NPU或训练。

**Status:** complete_static_recovery_package_not_executed

### STEP-216-A-RECOVERY-MF：双侧PGID异常清理补齐

- [x] Host异常/中断时若`$output/launcher.pgid`存在，先在容器内严格校验正整数且`>1`，精确TERM负PGID、有限轮询、KILL；文件不存在视为runner未启动。
- [x] 容器launcher清理后再清理宿主docker-exec PGID，trap保持到双侧清理与postflight结束。
- [x] 新增严格PGID语法及清理顺序合成测试，刷新source contract；静态8项测试、`bash -n`与diff-check通过。

## STEP-216-A-CORE：唯一world8核心测试与接口修复（2026-08-15）

- [x] 将b92c恢复包部署至业务repo外TOOL_ROOT；预检source/runtime identity、bash-n、HEAD/allocator-only、端口/训练/后8空闲并唯一启动world8。
- [x] 8 worker已创建，但rank0在ready前因真实SOAP bound method签名与错误静态合同不符fail-closed；其余7 rank统一TERM，0 core样本且不重跑。
- [x] postflight确认进程/端口/back8归零，永久保留failure summary/manifest。
- [x] 仅本地把project/project_back修为完整state接口，隔离old/baseline/candidate Q视图，并新增fake SOAP bound-method与AST门禁；未再次上传或运行。

**Status:** complete_runtime_rejected_signature_contract_fixed_static_only

## STEP-216-A-PID：修复后唯一core与PID namespace合同（2026-08-15）

- [x] 以773574 contract在新repo外TOOL_ROOT完成唯一world8：8 rank、559 state/543 factor ready及后8die live均成立，但PID namespace集合门禁失败，0 cycle且不重跑。
- [x] postflight确认rank/端口/back8归零，永久保留summary/manifest/live npu-smi。
- [x] ready字段改为明确`container_pid`；controller由npu-smi host PID读取宿主`/proc/<pid>/status`，取NSpid最后项映射container PID并保留host↔container↔die关系。
- [x] 容器init只读验证host PID的NSpid最后项为1；本地测试、source identity和bash-n通过，未上传修复包或启动NPU。

**Status:** complete_pid_namespace_contract_fixed_static_only

## STEP-216-A-HOSTCTRL：v3唯一core与宿主controller重构（2026-08-15）

- [x] 51cccd v3唯一world8达到ready8/559×543/live8，但容器内controller无法读取宿主`/proc/<npu_pid>`，0 cycle且不重跑；postflight归零并永久留证。
- [x] controller supervise迁移为host Python；容器runner只启动torchrun、写PID/PGID并等待host release，不再执行controller。
- [x] host launcher并行管理docker-exec runner与host-controller两套setsid PID/PGID，任一失败按container launcher→controller→docker-exec顺序精确清理并postflight。
- [x] 新增纯stdlib host-controller ready/proc/npu-smi/release/done fixture；静态12项、source identity、pycompile、bash-n通过。未上传重构包或运行NPU。

**Status:** complete_host_controller_static_ready_not_executed

## STEP-216-A：TurboSOAP Brockett + 单次 cubic polar 局部筛选包（2026-08-15）

- [x] 固定社区一手来源 TurboSOAP `1339218c...` / `soap.py` blob `d1563b35...`，以独立 JSON 冻结 FP32 core-probe 参数；任何来源、参数或 scope 变化在 ready 前拒绝。
- [x] 从既有 STEP-215 adapter 的真实 checkpoint state 原位枚举 559 个 state、活动 23 类/543 个 `GG/Q` factor，并按原 stable descending sort 重建 `power_iter`；5120、未知 shape、非 FP32、非连续、非方阵或 grad-enabled 一律回退 `linalg.qr`。
- [x] 实现 baseline `linalg.qr` 与 Brockett `eta=0.01`、单 substep、一次 scaled cubic polar retraction 的交替 Event/wall 局部 A/B；本阶段禁用 TurboSOAP 的 eigengap、EMA 和自适应 controller，只审计 core 算法。
- [x] 冻结 finite、Q 正交 max/Fro `<=2e-5`、重复性 `<=1e-5`、Rayleigh offdiag、真实 `exp_avg_sq` marginal 下一步预条件作用 `<=5e-3`、peak memory及 Event/wall 每周期净省均 `>227ms` 的 fail-closed 门禁。
- [x] 完成 world8/back8 controller、runner、host launcher和摘要器；Python `py_compile`、AST 合同、6项CPU策略单测与两层GNU Bash `-n`通过。
- [ ] 仅在主任务明确调度且远端资源/容器/SHA门禁通过后运行一次局部 world8；静态包不授权业务修改、训练或提交。

**Status:** complete（仅静态实现与本地验证）

## STEP-215：SOAP QR 数值等价门禁与分阶段验证（2026-08-15）

- [x] 用户明确授权：后续候选不再以raw Q/R逐位一致作为唯一门禁；历史STEP-199/214-O裁决保留，不追溯改写。
- [x] 完成官方语义/API审计：`geqrf + orgqr`与reduced QR数学合同一致，当前固定栈没有ready Q-only补丁。
- [x] 恢复24类历史真实shape/count和当前543次活动权重，完成仓库外world8局部A/B harness；所有shape计时前强制warmup。
- [x] 完成SOAP唯一QR调用的最小`geqrf+orgqr`补丁草案，以及baseline双跑/candidate、连续两周期与首周期resume的仓库外状态门禁设计；未改业务。
- [ ] 连接通道恢复后，在唯一容器`mapqr-leicheng`的后8 NPU运行24-shape局部门禁，记录逐shape数值、Event/wall、HBM和543次加权周期收益。
- [ ] 仅当24类全部满足finite、Q/R NRMSE、正交与重构硬上限`1e-5`，且加权净省`>227ms/cycle`时，最小接入SOAP并验证连续两个QR周期。
- [ ] 连续周期通过后，再做checkpoint/resume与30-step单变量A/B；短程通过后才允许876-step长期收敛验证和提交决策。

**Status:** ready_local_harness_blocked_remote_transport_before_launch

### STEP-215-B：SOAP QR 2025～2026官方实现/补丁只读检索

- [x] 恢复`planning-with-files`上下文并与STEP-089～100、STEP-199、STEP-214-O去重。
- [x] 检索Ascend pytorch/op-plugin、CANN算子文档、PyTorch QR实现/issue及Triton-Ascend/CANNBot官方来源。
- [x] 按当前torch_npu2.7.1/CANN8.3RC1判定可复用commit/API、数值门禁适用性与理论收益，形成分阶段GO裁决。
- [x] 将网页结论只写`findings.md`，并在`progress.md`、`操作步骤.md`记录操作与边界。

**Status:** complete_go_staged_numerical_gate_no_ready_q_only_patch

### STEP-214-I 官方MSDA/FP32 atomic实现检索（2026-08-15）
- [x] 只读核验与隔离环境相同tag的Triton-Ascend `v3.2.0rc4`官方源码；确认官方仓库包含FP32、多核、重复地址高冲突`atomic_add`验收，但wheel载荷本身不附带这些测试。
- [x] 交叉当前main/release-3.2.2的fully-indirect/discrete atomic新增测试与lowering提交；这些能力晚于rc4，冻结环境不得以升级方式采用。
- [x] 审计DrivingSDK ScatterAddV3的UB内聚合后GM atomic方案及后续精度/兼容修复；其重点优化为`tailLen==1`，与空间MSDA `channel=32`不匹配，且无目标shape量化benchmark。
- [x] 排除STEP-213已关闭的MSDA load-balance/embed补丁、generic op-plugin `scatter_add`和CANNBot工作流本身；均不能作为可直接替换当前空间FP32 MSDA的严格等价实现。
- [x] 裁决`NO_READY_FULL_MSDA_IMPLEMENTATION_GO_RC4_STRUCTURED_ATOMIC_MICROPROBE_ONLY`：官方rc4证据只开放静态重复索引FP32 atomic局部机制探针，不证明完整MSDA或`>22.7ms/step`收益。

### STEP-214-H Triton-Ascend FP32规则高冲突atomic局部门禁（2026-08-15）
- [x] 复用STEP-214-G已验证的world8后8设备、隔离venv、controller/release和全局环境快照合同；只新增仓库外atomic harness/runner/launcher，外层硬超时120s。
- [x] 同时测量rc4官方32-core重复地址c32模式，以及MSDA单batch等效`[576*8,32]`、32 programs×27 repeats=`864`贡献/output模式；禁止data-dependent/fully-indirect索引。
- [x] 8rank均完成finite、oracle exact、repeat exact、max_abs_diff=0；记录kernel-only和zero+kernel的NPU Event/wall时延、live npu-smi和release证据。
- [x] MSDA冲突型kernel Event为`2.38962ms`（8rank中位数的中位数），对应约`53.315G atomic-add/s`；按真实完整shape `14,092,861,440`次atomic吞吐外推约`264.33ms`，已经慢于DrivingSDK完整空间backward `146.580ms`。
- [x] 裁决`PASS_RC4_STRUCTURED_ATOMIC_MECHANISM_NO_GO_DIRECT_PER_CONTRIBUTION_MSDA_BACKWARD`：机制可用，但逐贡献GM atomic不是性能候选；只有能先做严格等价局部归并并显著减少GM atomic数的单一方案才可重开。

### STEP-214-J register局部归并FP32 atomic B1门禁（2026-08-15）
- [x] 对同一固定随机正负FP32输入比较direct 864次/output GM atomic与每program先register累加27项、仅32次/output GM atomic；PyTorch FP32双层sum为oracle。
- [x] 8rank B1 Event中位数由`2.437300ms`降至`1.207890ms`（`2.018x`），但B112线性外推`135.284ms`，未过严格`<123.88ms`门槛，故不运行真实B112。
- [x] aggregate相对oracle `max_abs<=3.814697e-5`、`NRMSE<=1.270072e-7`，优于direct；但两者atomic跨program顺序均非逐位重复，aggregate repeat max为`3.814697e-5`。
- [x] 裁决`NO_GO_REGISTER27_AGGREGATE_PERF_BELOW_22P7_AND_RAW_REPEAT_NOT_EXACT`；不进入fully-indirect、真实shape、完整MSDA或业务接入。

### STEP-214-K 空间MSDA forward B1真实shape Triton原型（2026-08-15）
- [x] 实现真实B1签名`value[1,576,8,32]`、`sampling[1,15360,8,1,8,2]`、8点bilinear/zero-padding/attention reduction，并准备同输入DrivingSDK oracle/Event合同。
- [x] gate1在首个Triton warmup前被rc4 host检查拒绝：grid/coreDim `122880>65535`；无ready、无正式时延/数值样本。
- [x] 只按runtime明确提示在隔离runner加入`TRITON_ALL_BLOCKS_PARALLEL=1`后运行唯一gate2；device runtime仍以`coreDim=122880 can't be greater than UINT16_MAX`拒绝launch。
- [x] 按gate2失败即收口合同裁决`NO_GO_RC4_COREDIM_LIMIT_TRUE_SHAPE_FORWARD_PROTOTYPE_UNEXECUTABLE`；不改kernel网格、不运行B112、不训练或接业务。

### STEP-214-L 空间MSDA forward双head/grid61440候选（2026-08-15）
- [x] 作为独立候选将grid改为`15360*4=61440`，每program静态处理相邻2 heads×32 channels；不设置`TRITON_ALL_BLOCKS_PARALLEL`，bilinear/zero-padding/attention数学不变。
- [x] 唯一world8 B1 gate通过数值：边界/越界随机坐标下`max_abs<=1.549721e-6`、`NRMSE<=2.742497e-7`，SDK/Triton均repeat exact/finite。
- [x] DrivingSDK/Triton B1 Event分别为`0.644240/2513.123901ms`；Triton约`3901x`更慢，远超`0.48159ms`准入线。
- [x] 裁决`NO_GO_TWOHEAD_GRID61440_EXTREME_PERFORMANCE_REGRESSION`；不运行B112、不训练、profile或改业务。

### STEP-214-M 空间MSDA forward Q-tiled32/grid3840候选（2026-08-15）
- [x] 每program处理32 queries×1 head×32 channels，8点/四邻域在program内静态归约，grid=`480*8=3840`；不复用过细grid。
- [x] 唯一world8 B1 gate数值通过：`max_abs<=1.490116e-6`、`NRMSE<=2.052884e-7`，SDK/Triton repeat exact/finite。
- [x] DrivingSDK/Triton Event为`0.609280/423.697617ms`，Triton约`695.4x`更慢，B112外推`47454.13ms`。
- [x] 裁决`NO_GO_QTILE32_EXTREME_PERFORMANCE_REGRESSION`；不运行B112、训练、profile或业务改动。

### STEP-214-N 空间MSDA persistent grid64候选（2026-08-15）
- [x] 核验rc4官方tag含`tl.range` pytest/tutorial/gather与scf.for lowering测试；动态循环前端/后端具备机制依据。
- [x] grid64、BLOCK_OUT256，以非展开`tl.range`按stride16384遍历flattened q/h/c，并在每块内完成8点×四邻域；不设置ALL_BLOCKS。
- [x] 唯一world8 B1 gate数值通过，但DrivingSDK/Triton Event为`0.643330/1518.993774ms`，B112外推`170127.30ms`。
- [x] 裁决`NO_GO_PERSISTENT64_EXTREME_PERFORMANCE_REGRESSION_CLOSE_TRITON_FORWARD`；按约关闭Triton forward，不再扩展tile sweep。

### STEP-214-O SOAP QR geqrf+orgqr局部原语门禁（2026-08-15）
- [x] 固定`[2560,2560]` FP32确定输入，world8交替7次比较`torch.linalg.qr(reduced)`与`torch.geqrf+torch.orgqr`；候选时延不含诊断性`triu`。
- [x] 两个API均支持NPU dispatch；Event由`4027.1255ms`降至`1262.1964ms`，约`3.191x`，但raw Q非bitwise。
- [x] Q `max_abs<=4.734844e-6/NRMSE<=3.960207e-6`；R `8.809566e-5/1.217481e-6`，正交/重构仅作诊断，不放宽raw-Q规则。
- [x] 裁决`NO_GO_RAW_Q_BITWISE_MISMATCH_DESPITE_3P19X_SPEEDUP`；不扩24类shape、不接optimizer或训练。

## 当前恢复点：GPU合同对齐单次全阶段profile已闭环（2026-08-14）

- [x] 客户GPU配置、seed0/deterministic=False、batch/rank=16与8卡NPU合同对齐。
- [x] profiler-off 30-step稳定基线；普通步NPU/GPU吞吐比约0.700，完整稳定周期约0.510。
- [x] 稳定step之后完成唯一一次rank0 `with_stack+record_shapes` 连续窗口，覆盖SOAP及普通步；不按阶段重复采集。
- [x] 同一trace完成全局/逐step TopN、异常JSON、10节架构报告、v4栈归因、候选原因/优化方式/决定矩阵。
- [x] 原始profiling及已摘要work/log精确删除，raw=0；HEAD `f922c38` clean，基线checkpoint保留。
- [ ] 仅在出现新的固定环境、状态/功能等价、独立可回退候选时进入算子A/B、8卡短训A/B及同checkpoint测试集门禁。
- [ ] 目标NPU8/GPU8达到1:1或更好；当前未达成，不得以改变最终功能、loss或optimizer状态换性能。

### 2026-08-14 本轮恢复审计

- [x] 确认用户更新后的主线仍为“以预热后的稳定 Step 单步 profiling Top N 推进”，与 STEP-189～192 现有证据链一致，不回退到冷启动耗时或未分组累计耗时排序。
- [x] 恢复当前唯一未完成动作：STEP-192 的后8卡、8-rank真实shape局部channels-last机制门禁；在门禁完成前不修改业务代码、不创建性能commit。
- [x] 记录工具错误：尝试新建 `/goal` 时发现本任务已有active goal，改为读取并沿用用户更新后的现有goal；`session-catchup.py` 返回exit 1且无诊断文本，改为直接读取计划文件首尾、Git状态和最近操作记录恢复上下文，未重复执行失败命令。
- [x] 记录本机运行时错误：系统`python.exe`为Windows Store占位程序，远端预检helper与本地helper自检均exit1且无输出；已停止使用该入口，改用Codex bundled Python并验证`paramiko 4.0.0`及helper `--help`正常。另记录一次JS模板字符串误解析shell变量导致的本地SyntaxError，以及本机旧.NET缺少`SHA256.HashData`；均在状态变更前失败，分别改为固定相对路径和`Get-FileHash`。

### Authoritative next step

基于 STEP-183 的 876-step 收敛/恢复证据和 STEP-185 的同 checkpoint 推理证据完成交付审计；当前不再为追求数字继续修改 MSDA。若客户提供不改变固定训练环境的 canonical OR-Tools 评测入口，再用已冻结合同补齐全量绝对指标；否则明确保留为外部依赖待办。

## STEP-186 最终交付审计与报告（2026-08-14）

- [x] 修正STEP-183/184中已由后续权威结果证明完成、但仍未勾选的历史状态；更新顶部`Next Step`与`Current Phase`。
- [x] 以永久算法基线`63861df`、可执行客户基线`4c37039`、当前`f922c38`和GPU日志形成统一30-step主表，指标名保持`time/memory/loss/grad_norm/throughput (samples/s)`。
- [x] 汇总DrivingSDK MSDA的876-step训练、checkpoint/resume、512-sample同checkpoint推理性能与固定shape数值门禁。
- [x] 审计采用commit链、DrivingSDK/通用候选关闭状态、profiling raw清理状态和未修改功能的证据边界。
- [x] 生成`最终性能优化报告.md`；未改远端业务代码、未创建commit、未启动训练。
- [ ] 客户提供既有canonical OR-Tools评测环境后，补齐25,287样本绝对RG指标；该项为外部依赖，不在固定容器安装软件。

**Status:** in_progress（交付报告完成；仅canonical全量绝对评测等待外部环境）

## STEP-187 本地一次性产物清理（2026-08-14）

- [x] 固化规则：临时文件、探针、profile、缓存、runtime config、work和中间结果在结论归档且后续无需复用后立即删除；保留客户输入、最终报告、脱敏证据和仍需使用的连接工具。
- [x] 盘点并验证清理目标均位于当前工作区内、不是工作区根目录、内部无reparse point；二次Git复核发现`.codex-remote-edit`含37个tracked夹具，立即恢复且status0，仅清理其中11个untracked临时文件。
- [x] 删除`work`、`.codex-tools`中65个已闭环候选的一次性顶层脚本、两处字节码缓存及`.codex-remote-edit`的untracked临时文件；净删除247个文件、1,737,330 bytes。
- [x] 保留`.codex-remote-edit`全部tracked夹具、`.codex-tools/remote_exec.py`、`remote_sync.py`和`python-packages`源码，因为后续只读连接和canonical评测环境核验仍需使用；保留`custom`、GPU日志、计划文档和最终报告。

**Status:** complete

## STEP-188 GPU/NPU最大公共step口径修正（2026-08-14）

- [x] 确认GPU日志完整步数3664、当前`f922c38`完整训练步数876，最大公共窗口为1～876。
- [x] 从本地GPU日志重新解析前876条`time/memory/loss/grad_norm`，验证索引876/876唯一且连续。
- [x] 将GPU/NPU主对比从前30步改为876-step；30-step仅保留为`4c37039`永久基线的历史同口径数据。
- [x] 更新最终报告及四份持久记录；本轮未创建临时文件、未启动训练、未占用NPU。

**Status:** complete
## STEP-189：普通步全算子基线（2026-08-14）

- [x] 完成当前 `f922c38` 普通训练步低开销采集（客户 batch16/rank、后 8 NPU、8 rank、8 step）。
- [x] 输出全部 243 个唯一算子及 `call_count/kernel_duration_total_ms/kernel_duration_avg_us/wait_time_total_us/total_cost_us/step_kernel_share_percent`。
- [x] 完成异常 schema、10 节架构报告、SHA 与 correctness gate。
- [x] 清理全部远端 profiling raw 与本地一次性脚本。
- [ ] 以 device underfeed 为第一优先级做 host wait/launch gap 细分；随后按 Conv、MSDA、Layout/TransData 的独立收益上限选择下一项功能优化。

**Status:** profiling_complete_next_candidate_pending

## STEP-190：MatMul HF32 单变量候选

- [x] 排除已闭环的DDP unused、MapTR索引族、pin、CPU affinity、internal format与Conv HF32方向。
- [x] 当前普通步证据：Attention/MatMul纯kernel=`224.836641 ms/step`，超过22.7ms准入线。
- [x] 用一次8-rank单步shape-only hook取得111类/133次调用；只记录module/shape/dtype/count，不记录张量值。前三类代表shape来自BEV sampling offsets、BEV FFN和lane3d output projection。
- [x] 后8卡8-rank执行MatMul HF32 False/True的forward/backward性能与数值门禁，Conv HF32保持False；三个shape合计中位耗时`15.002210 -> 13.738930 ms`，节省`1.263281 ms`、`1.091949x`，最坏NRMSE=`1.469314e-4`。
- [x] 裁决：按真实调用次数加权的已覆盖收益约`3.964 ms/step`，且全局HF32改变输出/梯度；未达到`22.7 ms/step`阶段门槛，拒绝进入客户1-step/30-step训练A/B，不修改业务代码、不commit。
- [x] 删除shape probe和HF32门禁的原始JSON、训练/launcher日志、work、临时config/hook/wrapper与远端传输目录；只保留脱敏摘要、Markdown/JSON、SHA和cleanup记录，远端Git clean、进程0、端口0。

**Status:** complete_rejected_before_training_ab

## STEP-191：Inplace ReLU 形状与严格等价候选审计

- [x] 复核高频`LinalgVectorNorm`：历史STEP-109/123已证明PyTorch默认路径自动使用foreach，显式`foreach=True`无收益，当前不重复测试。
- [x] 复核DDP unused、Nonzero/Index/Unique/IndexPut、SOAP zero/foreach、Conv HF32、全局internal format均已有必要性或正式拒绝证据。
- [x] 取得90次in-place与59次module out-of-place ReLU元数据；加权元素量分别`5,539,627,008/598,180,608`，前者约9.26倍。
- [x] 结合稳定Step7纯kernel耗时归一化：in-place/out-of-place约`0.006223/0.007714 ns/element`，in-place单位元素反而快19.33%，总耗时差异来自shape规模，不是in-place实现劣化。
- [x] 裁决：最大in-place激活`[112,64,288,512]`含`1,056,964,608`元素，改out-of-place需额外约4.23GB输出；无性能收益且增加峰值显存/alias风险，不进入机制或训练A/B，不改码、不commit。
- [x] 清理单步raw JSON/log/work/hook/config/wrapper/传输目录，只保留脱敏summary/JSON/Markdown/SHA/cleanup；远端Git clean、进程0、端口0。

**Status:** complete_rejected_before_mechanism_gate

## STEP-192：冻结图像 Backbone+FPN 局部 channels-last 门禁

- [x] 源码确认客户`fix_backbone=True`，训练主链对当前图像backbone+neck执行整模型`eval()`与`torch.no_grad()`；ReLU探针也证明主导图像激活`requires_grad=False`。
- [x] 候选与已拒绝的全局`allow_internal_format=True`分离：只在冻结图像backbone+neck边界转换channels-last，保持全局开关、Conv/MatMul HF32均为False，并在输出边界恢复contiguous NCHW语义。
- [ ] 后8卡8-rank对真实`[112,3,576,1024]`输入比较NCHW与局部channels-last的backbone+FPN forward；候选计时必须包含输入和最终输出layout转换。
- [ ] 比较4层输出shape/stride/finite/max_abs/NRMSE；仅收益超过22.7ms且数值/显存门禁通过才进入1-step训练。
- [ ] 新增GPU合同门禁：以本地根目录`mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py`为GPU权威配置，逐项核对远端生效配置；除NPU设备/算子适配所必需差异外，batch、数据、模型、优化器、调度、随机性、训练步和评测语义必须保持对齐。
- [x] 远端3项工作区变化已确认均为前序STEP-192诊断产物/运行副作用；精确恢复`fusion_result.json`并删除仓库内旧诊断脚本与传输目录后，Git status count=0。
- [x] GPU/NPU配置语义审计完成：共42项差异；8卡、batch16、LR、runner、DDP、checkpoint/evaluation一致，但数据、随机增强、`use_grid_mask`和optimizer hook存在功能差异。
- [x] 裁决：在严格GPU合同重建前暂停本候选；不启动channels-last 8-rank门禁、不修改业务代码、不commit。

**Status:** paused_by_gpu_contract_mismatch

## STEP-193：GPU权威配置合同重建

- [x] 以本地根目录GPU配置SHA`9039BD31...CA33B`为权威输入，在远端原位完成与NPU配置SHA`217EC2E7...B721`的结构化比较。
- [x] 确认一致项：`num_gpus=8`、`batch_size=16`、`lr_config`、`runner`、`find_unused_parameters`、`checkpoint_config`、`evaluation`。
- [x] 追溯并分类42项差异：路径镜像/插件适配、已批准等价性能实现、诊断日志差异、功能差异。
- [ ] 对训练/验证数据引用证明样本集合和顺序等价；仅路径不同不能直接认定等价。
- [x] 仓库外运行时配置已恢复：`dropout_sd_prob=0.2`、lidar`0.1/0.2/0.2`、`use_grid_mask=True`、GPU动态loss-scale FP16 hook；保留NPU路径/插件、已验收SOAP阈值和日志降频。
- [x] 结构化复核由42项收敛到31项；剩余差异不再包含模型随机项、grid mask、train dropout或optimizer hook，运行时配置SHA为`02ACA0C7...F56A5`，业务Git clean。
- [x] 使用冻结SHA`10AD92C7...E0FC` canonical harness完成后8卡、8-rank、batch/rank16动态FP16门禁；启动环境修正后1-step完成，按GPU oracle扩展4步并在step4恢复有限grad。4步loss相对GPU同编号约-0.01%/-1.67%/-0.64%/-1.07%，fatal/OOM0，原始产物已清理。
- [x] 在对齐合同上完成8卡seed0/deterministicFalse 30-step稳态基线：30/30、fatal/OOM0、iter4起grad finite；稳定普通step15～29排除24的mean/median/P95=`6.180/6.230/6.940s`、CV9.55%，SOAP step14/24=`29.579/29.222s`、相差1.21%。
- [x] 只采集一次稳定区间内的多Step连续窗口profiling：rank0启用`with_stack=True + record_shapes=True`，`wait+warmup`跨过step1～22，active锁定step23～26，同时覆盖稳定普通与SOAP周期；同一trace分析数据/Host、前向、loss、反向、optimizer、HCCL和空泡。带栈trace只用于算子排序/归因，端到端性能以无profiler基线为准。
- [ ] 每个TopN优化候选必须依次通过：真实shape算子输出/梯度A/B；8卡短训loss/grad与端到端性能A/B；同一checkpoint、同一测试集、同一样本顺序的输出与任务指标A/B。功能无回归且性能净收益超过噪声后才允许形成单一可回退commit。
- [x] 新TopN重新排序及全阶段栈归因已完成；channels-last、QR、MSDA、Conv/TransData、MapTR索引族等旧方向按证据保持关闭。

**Status:** in_progress

## STEP-194：稳定普通步 BMM/ViewCopy 新边界复核（2026-08-14）

- [x] 从已保留的v4栈归因和逐Step TopN确认BMM/ViewCopy的算子语义、源码调用点、普通步稳定性及纯kernel上限；不使用wait time制造收益。
- [x] 对照历史point_sampling BMM、SOAP view/copy/foreach及MapTR target路径的正式拒绝证据，避免把同名算子误当成新候选。
- [x] 在远端`f922c38` clean HEAD只读审计对应源码，寻找单一producer-consumer边界中的冗余expand/transpose/contiguous/view/copy或重复BMM。
- [x] 只有新的独立可回收上限超过22.7ms/step且不改变alias、stride、输出、梯度、loss或optimizer state，才创建仓库外真实shape机制门禁；本轮不存在满足条件者，因此不创建门禁。
- [x] 无新边界，形成`CLOSED_NO_NEW_BOUNDARY`证据，不占用NPU、不改码、不commit。
- [x] 用户新增偏好已登记：未来profile在当前分析仍需复用期间暂不删除；本轮16.65GB raw在该要求到达前已按旧规则删除且不可恢复，不为补回数据重复采集。

**Status:** complete_closed_no_new_boundary

## STEP-195：冻结图像Backbone+FPN Conv-BN推理折叠

- [x] 证明客户活动路径在图像Backbone+FPN调用期间确实为`eval()+no_grad()`，且训练阶段不会更新BN running stats或Conv/BN参数。
- [x] 枚举活动ResNet34和FPN中的直接相邻Conv-BN对，排除残差分支、共享模块、非相邻或训练态BN；用稳定profile的BNInfer纯kernel量估算可回收上限。
- [x] 设计checkpoint/state_dict兼容边界：原注册模型保持权威，checkpoint加载后创建未注册eval-only融合副本；state/hash、参数/缓冲区及optimizer ID门禁通过。
- [x] 在后8逻辑NPU、8 rank、真实`[112,3,576,1024]`输入完成四层shape/stride/finite/max_abs/NRMSE、峰值显存和完整边界device计时。
- [x] 数值门禁失败后严格停止：虽然边界节省51.058ms，但最大NRMSE=`1.991e-3`超过`1e-4`，不进入短训/测试集A/B、不改业务、不commit。

**Status:** complete_rejected_numeric_gate

## STEP-196：稳定普通步Host/Device underfeed现有证据复核

- [x] 仅复用STEP-189及唯一全阶段profile的脱敏报告，拆分host launch gap、Python/C++调度、同步点、DDP、数据等待及Index/Reduction；未重新采集profiling。
- [x] 与历史必要性/正式拒绝路径交叉：未发现源码边界唯一、状态等价且理论净收益超过22.7ms/step的新对象。
- [x] 固化约5.85s分散underfeed及Index/Reduction外层聚合的证据缺口；不凭猜测改代码、不训练、不重复profile。

**Status:** complete_closed_no_unique_boundary

## STEP-197：冻结图像Conv-BN选择性分组折叠门禁

- [x] 将43对按`stem/layer1/layer2/layer3/layer4/FPN`拆为`1/6/9/13/7/7`对；同checkpoint CPU严格加载258键并验证原state hash、参数/缓冲区ID不变。
- [x] 仅在完整名称`mapqr-leicheng`中以后8逻辑NPU、8 rank、确定性真实`[112,3,576,1024]`输入完成6个单组门禁；不训练、不profiling、不安装依赖、不改tracked业务代码。
- [x] 每组逐一创建并释放融合副本，比较四层shape/stride/finite/max_abs/NRMSE及完整Backbone+FPN NPU event时间；六组结构门禁均通过。
- [x] 六组最大NRMSE均超过`1e-4`，且净节省均低于`22.7ms`；按预声明规则不测试组合、不进入短训/测试集A/B、不commit。
- [x] 裁决`REJECT_NO_SELECTIVE_GROUP_MEETS_NUMERIC_AND_22P7MS_GATE`；精确删除逐rank JSON、日志、PID、选择文件和一次性脚本，仅保留两份脱敏summary。

**Status:** complete_rejected_numeric_and_performance_gate

## STEP-198：稳定TopN的图执行/下发融合固定环境可行性审计

- [x] 只读核验固定容器现有`torchair/torch.compile/torch_npu`图模式与NPU Graph能力：TorchAir缺失、backend npu退化eager，原生NPUGraph可用。
- [x] 对照历史计划与A/B，排除已经正式拒绝的TASK_QUEUE/COMBINED、全局internal format、HF32和同类重复实验。
- [x] 确认最小独立边界为冻结图像`img_neck(img_backbone(img_flat))`；原算子数学、checkpoint和模式切换保持不变，权威仓库/配置纠正复核通过。
- [x] 在完整名称`mapqr-leicheng`中以后8逻辑NPU、8 rank、同iter30 checkpoint和确定性真实`[112,3,576,1024]`输入完成原生NPUGraph机制门禁；不训练、不profiling、不改业务tracked源码。
- [x] 手工capture使用API要求的独立NPU stream；11次eager与11次graph warmup后同进程交替8轮，graph计时包含动态输入`copy_`和`replay`，输出直接使用static outputs。
- [x] 四层输出shape/dtype/stride/finite完全一致，最大绝对误差、NRMSE和重复replay误差均为0；state SHA、参数/缓冲区及optimizer引用保持不变。
- [x] 性能门禁失败：eager中位326.609ms，graph(copy+replay)中位327.281ms，净节省-0.714ms、0.99795x；capture额外reserved约30.667GB，裁决`REJECT_MECHANISM_GATE_NO_TRAIN_NO_COMMIT`。
- [x] 三轮仓库外机制尝试及前两次harness错误均已记录；第三轮exit0。远端raw 21文件和harness 2文件精确删除，仅留脱敏summary/manifest；Git clean、进程0、端口0、profile raw0。
- [x] 限制：第三轮已实时确认8 workers、rank0～7/local_rank0～7和端口监听，但宿主/容器`npu-smi`采样均未返回对应PID，严格live PID证据缺失；本候选仍因收益为负而关闭，不重跑粉饰证据。

**Status:** complete_rejected_performance_gate

## STEP-199：SOAP 周期 QR 项目内自定义等价算子可行性审计

- [x] 只读清点固定容器现有自定义算子工具链、编译器、头文件与可调用 QR/Householder API；禁止安装、升级或替换客户依赖。
- [x] 固化 SOAP QR 的算法、Q 符号、排序、状态更新、投影、dtype 与 checkpoint 恢复等价合同，并与 GPU 权威路径对齐。
- [x] 交叉排除 STEP-089～100 已拒绝的 batched QR、geqrf/orgqr、out-buffer 和 multi-stream 方案，避免换名重试。
- [x] 固定环境仅有当前已使用且强制同时输出Q/R的`aclnnLinalgQr`，无同语义Q-only或更快primitive；自写AscendC/TBE会改变算法与raw Q状态轨迹，形成NO_GO，不进入机制门禁。

**Status:** complete_no_go_fixed_environment

## STEP-200：稳定 Step 亲和 API TopN 复核

- [x] 复核当前稳定 profile 摘要中 `torch.addmm` 建议：活动路径已由`nn.Linear→F.linear`降为单个`aclnnAddmm`，117次调用分散于BEV/decoder多个动态边界；稳定Step7纯kernel/含wait总上限仅`15.089/19.865ms`，均低于`22.7ms`，裁决`NO_GO_ALREADY_FUSED_AND_BELOW_THRESHOLD`。
- [x] 复核 `torch_npu.npu_confusion_transpose` 建议：当前稳定普通Step中可勉强相关的`InplaceCopy_Transpose+Contiguous_Transpose`纯kernel合计仅`21.941ms`且跨消费者；SOAP活动路径无可融合的活跃reshape-transpose对，固定A3环境该API又只有`acl_op`实现而无ACLNN实现，裁决`NO_GO_UNSUPPORTED_AND_BELOW_THRESHOLD_NO_UNIQUE_BOUNDARY`。
- [x] 复核 `torch_npu.npu_add_layer_norm` 建议：仅BEV 3处和MapTR decoder 12处为直接邻接；LayerNorm前后向纯kernel仅`6.636ms`，把每步全部Add与全部LN-forward错误假设为可消除的极端上限也仅`21.007ms`，裁决`NO_GO_NO_UNIQUE_ADD_LAYERNORM_BOUNDARY_ABOVE_22P7MS`。
- [x] 三类建议均按当前 torch_npu 2.7.1、四时钟、跨流 busy union 与 wait-anchor 规则完成裁决；没有单一严格等价边界净收益上限超过 22.7ms，因此不创建机制门禁。

**Status:** complete_no_go_below_threshold_or_unsupported

## STEP-201：1:1目标固定环境阻塞审计

- [x] 复核唯一稳定全阶段profile的普通步、SOAP周期、underfeed、TopN、亲和API及全部采用/拒绝矩阵；没有遗留的单一严格等价边界。
- [x] 确认目标仍未达到：稳定普通步NPU/GPU吞吐比约`0.700`，完整稳定周期约`0.510`。
- [x] 确认继续实施所需条件均超出当前授权或证据：厂商提供同语义更快primitive、客户允许固定软件栈能力变化，或授权在原raw已不可恢复后采集新的稳定timeline以唯一归因分散underfeed。
- [x] 连续三个目标轮次均复现同一固定环境/证据阻塞；不以改变loss、SOAP数学、optimizer/checkpoint状态、batch或GPU对齐合同绕过。

**Status:** blocked_external_capability_or_new_evidence_required

## STEP-202：获准重新采集一次稳定全阶段 profiling

- [x] 复核权威仓库`ascend_npu_optimize@f922c38`、GPU对齐运行时配置/checkpoint、正确容器、后8逻辑NPU及旧profiling harness合同；全部SHA和配置静态门禁通过。
- [x] 固定唯一采集：8 rank、batch/rank16、seed0/deterministic=False，rank0 `with_stack=True + record_shapes=True`，`MAX_ITERS=28`、schedule=`22/1/4`，从头训练且checkpoint关闭；所有阶段从同一trace分析。
- [x] 启动前/运行中/结束后核验`torch_npu`、8 rank、`npu-smi` 8个对应进程、端口、fatal/OOM、loss/grad有限及Git状态；唯一任务28/28自然exit0，tracked运行副作用已精确恢复且不触碰诊断目录。
- [x] 远端原位生成逐step四时钟、TopN纯kernel/total-cost双排名、wait-anchor、AICPU暴露、bubble前后kernel/host证据、全阶段栈归因、异常JSON与独立10节架构报告；schema校验0错误、架构10/10节、40项权威分析工件SHA校验通过。
- [x] 本轮原始profiling按用户最新明确要求永久原位保留，后续分析/候选完成后也不删除、不移动、不覆盖且不拉取本地；205个文件已全量SHA，树外manifest验证`deletion_authorized=false/retained=true/mutation=false`。
- [x] 按新的稳定TopN完成候选筛选与门禁：P0 `random_spatial_mask`在短窗加速但被876-step长期A/B反证；P1 MSDA差距位于固定SDK单kernel；STEP-206确认其余TopN没有新的单一越线边界。

**Status:** complete_profiled_analyzed_and_candidates_closed

## STEP-203：GPU 无堆栈 profiling 对照 NPU 稳定全阶段 profiling（2026-08-14）

- [x] 安全盘点 GPU `.pt.trace.7z`：普通文件、非 symlink、bytes、SHA256、可读性、远端既有解压工具；使用系统既有`libarchive.so.13`安全解包，不覆盖归档、不安装环境。
- [x] 在新的远端诊断目录原位解包并识别真实 trace schema、Profiler step/捕获范围、稳定普通 step、SOAP 边界及 GPU 配置合同；不假设 GPU/NPU step 编号一致。
- [x] 建立跨设备语义对齐合同：用稳定普通 step 模板、调用次数、shape/layout/dtype 与业务边界映射 CUDA kernel 家族到 ACLNN/NPU 算子族；GPU 无堆栈的源码归因降级标注。
- [x] 输出 GPU baseline、NPU/GPU 四时钟和阶段 ratio、同义算子族 TopN 差距、NPU 独有超额、host launch/underfeed/H2D/同步及理论可回收上限。
- [x] 与历史采用/关闭矩阵交叉复核，形成按优先级候选、优化原因/方式、等价风险和算子→短训→同 checkpoint 测试集门禁。
- [x] 全部远端原位分析；GPU 归档与本次 NPU raw/manifest 永久保留，不删除、不移动、不覆盖、不拉取本地；本地只保存脱敏统计和操作记录。

**Status:** complete_p0_random_spatial_mask_mechanism_gate_pending

## STEP-204：`random_spatial_mask` 批量低分辨率 mask 严格等价优化（2026-08-15）

- [x] 只读审计真实源码、调用条件、客户有效配置、真实 shape/count、CPU/NPU RNG 顺序、输出/alias/stride 和非整除边界；与 STEP-179 PillarVFE、STEP-194 ViewCopy/BMM 历史边界交叉去重。
- [x] 仓库外 CPU 与后8 NPU/world8 机制门禁：保持 enable 先行及每 batch 一次 CPU `randperm`，一次 H2D+`index_fill_`后在低分辨率网格展开；覆盖 512 个逐位/RNG/alias case，8/8 rank exact，完整同步边界净省 `206.246～235.591ms`，保守纯kernel下界仍超过`22.7ms`。
- [x] 独立 live 探针补齐8 direct rank、WORLD_SIZE/local_rank、torch_npu及宿主`npu-smi`物理4/0～7/1的8唯一PID证据；探针不重复机制、不训练、不profile。
- [x] 单文件最小业务 patch、CRLF一致性、py_compile、CPU真实业务函数64 case及8-rank NPU真实业务函数512 case exact门禁通过；当前未提交、可独立回退。
- [x] 旧30步raw日志已清理且无SHA，按裁决以仓库外detached`f922c38`重跑fresh baseline，再完全释放后以权威单文件patch跑candidate；两轮均30/30、exit0、8-rank/back8 live证据与永久日志/checkpoint SHA完整。首次baseline在import前因漏PYTHONPATH失败0 iter；保留失败证据后仅补历史成功前缀并用单一只读`.so`链接补齐detached运行时。
- [x] fresh paired统计：稳定普通14/14步均加速，mean`5.322551→5.000786s`、净省`321.765ms/6.045%`、吞吐`+6.434%`；cycle`-2.951%`，SOAP均值`+0.385%`属未改善。loss最大相对偏差`0.3934%`、grad最大`2.0626%`，dynamic loss-scale相位一致；checkpoint/state/optimizer schema、shape、dtype、finite一致但自然随机状态并非逐位。
- [x] fresh baseline/candidate各自resume连续性通过：均记录Iter30～36，loss全finite、Iter31 grad inf/scale下降相位一致，meta29→35、optimizer step26→32、schema/shape/dtype/finite一致。
- [x] 同GPU对齐合同fresh paired 876-step长期A/B完成：两边均876/876、exit0、loss全finite、动态scale相位一致；candidate相对baseline稳定普通慢`1.0719%`、SOAP快`1.5830%`、完整周期慢`0.0811%`、all1～875慢`0.0964%`、末100普通慢`2.2777%`，反证30-step短窗收益不可持续。
- [x] 固定512不再执行：STEP-185同一镜像身份仍不可恢复，但长期性能已先行触发拒绝，故数据门禁不再阻塞本候选裁决；不猜数据、不换测试集。
- [x] 裁决`REJECT_LONG_RUN_NO_SUSTAINED_NORMAL_OR_CYCLE_GAIN_NO_COMMIT`：不做876后resume、不commit；最终单文件diff以SHA=`921d53da...0313`永久保留后，仅精确恢复`bev_encoder.py`到HEAD，权威仓库与baseline tracked均clean。

**Status:** complete_rejected_long_run_no_commit

## STEP-205：DrivingSDK MSDA forward/backward 残余差距只读审计（2026-08-15）
- [x] 复用永久保留的 NPU 全阶段 profiling 与 GPU 无栈 trace，未重采集、未训练、未删除或移动 raw。
- [x] 按同义调用位置、shape、次数和 dtype 对齐 6 次 forward/backward；确认最大差距来自第 2 次空间 FP32 调用。
- [x] 分解项目侧 Cast/type_as/zero/launch 边界；可见纯 device 乐观上限约 6.236 ms/step，低于 22.7 ms 门槛，且无 TransData/AICPU/同步可消除。
- [x] 核验 SDK Python/_C schema、ACLNN headers、910_93 config 与同包算子：运行时未暴露 im2col_step、tiling、workspace、layout 或 precision 参数，也无同语义替代 primitive。
- [x] 交叉 STEP-079、STEP-138、STEP-146~152、STEP-183/185，避免重复迁移、zeros_like 或已采用融合。
- [x] 裁决 `NO_GO_MAIN_KERNEL_FIXED_SDK_NO_PROJECT_CONTROLLED_EQUIVALENT_BOUNDARY`；无业务改码、无 NPU 执行、无 commit。

**Status:** complete_no_go_main_kernel_fixed_sdk

## STEP-206：剩余TopN与GPU基线单一边界关闭审计（2026-08-15）

- [x] 复用STEP-202永久保留NPU raw/分析产物与STEP-203 GPU trace/JSON；不训练、不调用NPU、不重profile、不改环境、不删除或拉取。
- [x] 从普通Step24～26纯kernel TopN、total-cost wait-anchor、bubble、AICPU/host-self与GPU同义族向下交叉全部历史状态。
- [x] 完成AI-core ViewCopy保守上界和旧JSON字段纠错；random mask外无单一边界超过`22.7ms/step`。
- [x] 完成prelaunch scatter/H2D shape/count/GPU对齐；双方457次/step结构一致，历史pin路径已反证。
- [x] 完成forward+loss Matmul/BMM最内层源码拆分；唯一越线的`point_sampling`为历史正式拒绝项，其余新单点低于门槛。
- [x] 输出四时钟、count、纯kernel/wait、shape/stack/consumer、GPU对齐、历史状态、理论上限、风险与证据缺口关闭矩阵。
- [x] 裁决`CLOSED_NO_NEW_SINGLE_EQUIVALENT_BOUNDARY_ABOVE_22P7MS_AFTER_GPU_BASELINE_CROSSCHECK`；不创建候选代码或commit。

**Status:** complete_closed_no_new_single_boundary

## STEP-207：当前目标缺口与重开条件（2026-08-15）

- [x] 目标仍未达到：现有固定GPU合同下，NPU 8卡与GPU 8卡完整周期性能比仍明显低于`1:1`；不以改变loss、SOAP数学、batch、数据合同或checkpoint状态规避。
- [x] 新采集、GPU对照、P0长期A/B、DrivingSDK MSDA及剩余TopN全部闭环；当前固定软件栈内没有新的单一、严格等价、可独立回退且理论净收益超过`22.7ms/step`的项目可控边界。
- [ ] 仅在以下任一条件出现时重开：厂商在当前兼容栈提供同语义更快的MSDA空间FP32/QR primitive；客户授权经独立环境验收的软件栈能力变化；或恢复同一固定测试集身份并出现新的低扰动、可唯一归因证据。

**Status:** awaiting_external_capability_or_new_authoritative_evidence

## STEP-208：1:1目标完成度与系统级剩余边界复核（2026-08-15）

- [x] 按用户原始目标逐项审计当前权威证据，区分已完成、被反证、证据不足及真正外部阻塞，不以STEP-206关闭矩阵替代1:1验收；完成独立审计报告。
- [x] 从GPU/NPU稳定普通步及完整周期重新核对阶段差距，覆盖prelaunch/H2D、forward+loss、backward、optimizer/SOAP、通信和tail；没有新的项目可控单一越线边界。
- [x] 独立复核固定torch_npu/CANN/DrivingSDK环境的API、运行时、stream、allocator和compile能力；补齐`PER_STREAM_QUEUE=1`但当前不满足其多线程多compute-stream Dequeue准入条件。
- [x] 两路独立裁决均为NO_GO；形成第二次连续外部能力/权威证据阻塞，并给出schema兼容MSDA、逐位Q-only QR、非HCCL Dequeue直接证据或具体可回退软件栈方案四类精确重开输入。

**Status:** complete_second_external_blocker_recurrence

## STEP-209：第三次外部阻塞复核与目标状态收口（2026-08-15）

- [x] 只读复核正确容器、权威HEAD、固定软件栈、训练/NPU进程与永久保留NPU/GPU原始数据；外部状态未发生可重开变化。
- [x] 独立复核当前`0.583544:1`完成度、`3.222591s/step`缺口及现有授权范围；不存在尚未执行的单一、严格等价、可回退且`>22.7ms/step`候选。
- [x] 同一外部阻塞连续第三次成立；已记录精确重开输入并按目标规则将goal正式标记为`blocked`，未误报为完成。

**Status:** complete_third_recurrence_external_blocked

## STEP-211：环境变量组合适用性与历史A/B复核（2026-08-15）

- [x] 核对`expandable_segments`、ATB workspace两项、`TASK_QUEUE_ENABLE=2`和`TORCH_DEVICE_BACKEND_AUTOLOAD=0`的官方语义、默认值、版本与风险。
- [x] 交叉当前项目活动算子、显存/OOM状态、torch_npu导入路径及既有8卡正式A/B，判断每项是否实际生效。
- [x] 给出是否可直接加入客户训练合同的裁决：禁止整组加入；TQ2维持正式拒绝，ATB/allocator无适用性，autoload0仅保留于已验证入口。
- [x] 将网页来源、只读源码核验、历史数据和操作错误完整记录到`findings.md/progress.md/操作步骤.md`。

**Status:** complete_rejected_as_combined_performance_change

## STEP-212：GPU/NPU差距的社区与Triton自定义算子候选筛选（2026-08-15）

- [x] 以STEP-203/205同shape差距为输入，优先检索DrivingSDK/CANN社区中MSDA空间FP32、QR及同义算子patch，不以泛化环境变量替代算子证据。
- [x] 对照CANNBot Triton-op-generator支持范围，判断热点是否可表达完整forward/backward、动态shape、AMP及checkpoint严格语义。
- [x] 仅对具备固定环境工具链、单一边界和理论净收益`>22.7ms/step`的候选进入仓库外机制门禁；本轮没有同时满足条件者，因此未编译、未启动NPU。

**Status:** complete_no_go_current_sdk_already_contains_patches_and_no_ascend_triton_backend

## STEP-213：DrivingSDK v7.3后MSDA负载均衡补丁回移资格审计（2026-08-15）

- [ ] 获取并逐行解析v7.3之后910B/910_93 MSDA embed、cube/load-balance与精度补丁，确认空间FP32真实shape/key11实际经过的新代码。
- [ ] 在固定`CANN8.3.RC1 + torch_npu2.7.1 + mx_driving branch_v7.3.0`中核对项目局部新op名构建、隔离加载和独立回退机制，不安装或覆盖客户组件。
- [ ] 从永久STEP-202 raw检查空间MSDA kernel的task/core/subtask证据，只有可量化负载不均或目标shape官方benchmark才计算理论净收益。
- [ ] 仅当forward/三梯度/AMP/checkpoint严格语义、固定环境ABI和净收益`>22.7ms/step`同时成立时，设计8卡前的仓库外机制门禁。

**Status:** in_progress

## STEP-214：allocator正式加入、TaskQueue2重测与容器内隔离Triton-Ascend（2026-08-15）

- [x] 将`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`加入权威8卡启动入口，保持为独立可回退功能，不与其他性能变量混合；用户明确要求保留该单行patch，当前未commit。
- [x] 冻结同HEAD/同config/同seed的30-step allocator单变量A/B：candidate 30/30、exit0、8 rank/back8与环境继承门禁通过；普通/SOAP/周期分别回退`2.0135%/0.7971%/0.9968%`，因此裁决`NO_GO_PERFORMANCE_REGRESSION`，产物永久保留。
- [x] 在完整名称`mapqr-leicheng`容器内以隔离venv安装CANN8.3RC1兼容的Triton-Ascend3.2.0rc4；dry-run仅安装一个无依赖wheel，默认Python仍为全局Triton3.7.1，torch/torch_npu/CANN未覆盖。
- [x] 先验证隔离backend import和A3最小算子，再用于QR/MSDA自定义算子机制研究；安装成功不等于性能采用。
- [x] STEP-214-G最终以world8后8物理die完成官方vector-add A3最小机制门禁：8rank编译/加载、raw exact、live npu-smi与默认全局环境未变均通过；仅证明工具链，不等于QR/MSDA候选采用。
- [x] 隔离backend import已通过且registry仅含`ascend`；按资源协调要求尚未运行A3/NPU最小算子，等待allocator正式8卡A/B结束并由主任务明确释放资源。
- [ ] `TASK_QUEUE_ENABLE=2`另做单变量8卡A/B，不与allocator或Triton候选同轮测试；合同已冻结。一次启动与用户“算子局部对比优先”的新指示发生竞态，已在0 iter初始化阶段精确终止并永久保留暂停证据，不计有效A/B，等待后续恢复授权。

### STEP-214-C allocator-only正式30-step A/B
- [x] 复用STEP204 fresh baseline、冻结wrapper/entry/config/launcher及GPU合同；candidate仅注入`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`，并显式保持`TASK_QUEUE_ENABLE`缺席/default1。
- [x] 运行中确认8个direct rank、`WORLD_SIZE=8`、`torch_npu`加载、后8 chip的8个唯一PID；30步loss全finite、iter4～30 grad全finite、动态loss-scale轨迹一致，launcher自然`rc=0`。
- [x] 正式窗口：normal `5.322551→5.429718s`，SOAP `28.440240→28.666940s`，cycle `7.689581→7.766228s`；均无性能收益。训练/端口/NPU PID已释放，log/JSON/checkpoint/metrics/SHA永久原位保留。
- [x] 显存max/last由`25837→25698MiB`，降低139MiB；normal窗口max由`25652→25503MiB`，降低149MiB。checkpoint内容树的meta、shape/dtype/finite及scalar合同一致，但自然非确定性下tensor数值不逐位，不能据文件SHA或tensor逐位差宣称等价。

### STEP-214-B入口定位阶段结论
- [x] 已确认正式8卡环境继承链、AUTOLOAD0既有位置，以及tracked仓库中TQ/allocator当前未设置。
- [x] 已冻结allocator与TQ2分离的两轮单变量A/B合同；allocator永久patch只允许修改`tools/ddp_train.sh`一个export行。
- [x] 本地无权威脚本副本，历史诊断wrapper不得污染；用户补充授权后已在远端权威`tools/ddp_train.sh`落地唯一allocator export行并完成语法/diff/SHA门禁，未加入TQ2、未训练/NPU/profile。

### STEP-214-D SOAP QR最小Q-only候选CPU/源码设计

- [x] 复核权威`soap.py`、24类shape/count、stable sort和optimizer/checkpoint状态合同；将候选限制为单行QR边界，不聚合其他consumer。
- [x] 审计Triton-Ascend3.2.0rc4的FP32归约/dot、片上容量与核间同步接口；确认大shape必须GM tiled、多kernel panel，扩展同步不足以直接假设grid-wide barrier。
- [x] 计算当前R输出写带宽上限和每10步加权门槛：仅省R远低于`227ms/cycle`，收益必须来自整体AI Core实现。
- [x] 输出G0～G5最小harness/性能/状态门禁；raw Q任一点不逐位即早停，诊断容差不构成准入。当前裁决`NO_GO_FORMAL_RAW_Q_EQUIVALENCE_UNPROVEN_DESIGN_ONLY`，未编译/NPU/训练/业务修改。

### STEP-214-G Triton-Ascend A3最小机制门禁

- [x] 在仓库外准备官方vector-add world8 harness，完成py_compile、bash语法、源码SHA、rank/back8/npu-smi、全局不污染及精确回退合同。
- [x] 对首轮`算子exact但controller握手超时`保留失败证据，不误报PASS；修复controller前先完成CPU/file协议和finally-release自测，并维持kernel源码不变。
- [x] 唯一最终重跑在120s硬截止内完成：8 ready/8 done/0 failure、逻辑0～7/物理Phy-ID8～15、raw exact、live npu-smi、release和global before/after一致。
- [x] 结束后训练/torchrun/profile进程0且npu-smi全空；裁决`PASS_TRITON_ASCEND_WORLD8_BACK8_VECTOR_ADD_EXACT_GLOBAL_UNCHANGED`，未启动MSDA/QR或端到端训练。

**Status:** in_progress
