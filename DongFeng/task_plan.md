# Task Plan: Ascend NPU 性能优化

## Goal
在不破坏已完成 loss 对齐的前提下，为 `asend_npu_optimize` 分支制定并执行可复现、可量化、分提交的昇腾 NPU 性能优化方案。

## Next Step
只读调查本地规则、目标分支、loss 对齐提交、随机性固定代码、CPU fallback 与现有测试入口。

## Current Phase
Phase 1：计划已获用户批准，开始远程现状审计。

## Plan Status
**APPROVED — 2026-08-11**

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

## Decisions Made
| Decision | Rationale |
|---|---|
| 计划批准前设置硬门禁 | 用户明确要求不得直接修改代码 |
| 随机性移除单独成提交 | 遵守指定提交信息并隔离行为变更与性能变更 |
| 先基线/profile、后优化 | 避免凭静态印象优化，确保收益可量化 |
| 算子替换优先官方和成熟社区实现 | 降低正确性、维护性与版本兼容风险 |
| 每项性能优化独立验证和提交 | 便于定位收益、回归和回退 |

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| 暂无 | 0 | - |

## Notes
- 远程机器信息必须从本地 `机器IP.md` 读取，不在规划文件或回复中复制凭据。
- 本计划在 Phase 1 调查完成后会细化为文件/算子级候选清单；仍需用户明确批准才能实施。
