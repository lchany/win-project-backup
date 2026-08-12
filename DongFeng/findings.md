# Findings & Decisions

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
