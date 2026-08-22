# MX QrV2 端到端问题交接文档

更新时间：2026-08-22  
当前状态：尚未取得修复候选的有效 30-step 端到端结果；禁止宣称 MX QrV2 已修复。

## 1. 交接结论

本项目要解决的不是 standalone QR 数学测试，而是自定义 MX QrV2 在真实训练执行上下文中出现非有限值、loss 偏差或性能回退的问题。

历史证据已经证明：原 MX QR 在单算子测试中可以 finite、可重构，但端到端训练仍会出现 NaN。因此：

- standalone PASS 只是一道基本语义回归门禁；
- 后 8 卡、8 rank 的真实端到端训练才是修复判据；
- 当前 delta2-only 候选只通过了 standalone 门禁；
- STEP393 attempt1～6 均未产生可用于裁决 MX QR 的 30-step loss/耗时结果。

在当前启动链阶段，接手后不要继续增加 shape 扫描、单算子重放或 profiler。先把 STEP393 启动链精确推进到 8 rank ready，再运行一次低扰动 30-step。

### 1.1 项目背景

业务训练在 Ascend NPU 上使用 SOAP 优化器。SOAP 的两个固定调用点会调用自定义 `mx_driving_cloud.linalg.qr`，也就是本文所称的 MX QrV2。现场问题具有明显的上下文依赖：相同 shape 的独立 QR 测试可以通过，但真实多卡训练中曾出现 Q/R 非有限值、loss 偏离 GPU 基线或训练性能回退。

最初排查围绕三个问题展开：

1. 输入 A 是否本身含 NaN/Inf，或 192×192 FP32 shape 是否非法。
2. MX QrV2 的 QR 数学语义是否与 CPU 官方 QR 不一致。
3. 问题是否只在训练的异步执行、stream、内存复用和调用时序中出现。

已抓取的历史训练输入证明 A 有限；同一输入在 CPU FP32/FP64 QR 中正常。standalone 测试又证明原 MX QR 也可能返回 finite 且可重构的 Q/R。因此排查重点从“输入或公式错误”转向“真实训练上下文中的算子执行与状态”。

### 1.2 之前在做什么

前期工作分为四条线：

- 训练现场取证：抓取真实 192×192 FP32 输入及 Q/R，比较非有限区域、重构、正交、R 下三角和 CPU 官方 QR 投影。
- 源码诊断：分析 QrV2 的 GEQRT/LARFB/CalcQ 调用、T/V 生命周期和 Matmul position 声明，制作多个单变量候选。
- 候选构建：在不安装、不覆盖客户算子的前提下，构建独立 identity 的 diagnostic object，并通过 shadow OPP 优先加载。
- 端到端门禁：建立 STEP393 低扰动训练夹具，禁止逐 QR dump/profile/sync，目标是直接取得 30-step loss 与耗时。

### 1.3 本轮工作的目的

本轮不是继续证明“QR 单算子能算”，而是回答两个 release 问题：

1. delta2-only 候选在真实 8-rank 训练中是否消除 nonfinite，并保持 30/30 每步 loss 偏差不超过 2%。
2. 在语义和 loss 通过后，fixed MX 相对 fresh original MX 是否没有不可接受的性能回退。

### 1.4 当前正在做什么

当前正在把 STEP393 推进到第一次有效的 8-rank 端到端运行。attempt6 已经通过 source、canonical config、base 文件和 SOAP AST 静态门禁，但在 environment preflight 的 `torch/torch_npu` 导入链发生 ImportError，尚未创建 rank。下一项工作是取得精确 ImportError、改进 failure evidence 和 host waiter 早停，然后只做一次最小启动链验证。

## 2. 最终验收门禁

### 2.1 正确性

- 必须使用完整名称为 `mapqr-leicheng` 的现有容器。
- host 分配/可见设备必须是后 8 张 Ascend NPU，即设备编号 8～15。设置该可见集后，进程内逻辑 device 通常重新编号为 0～7；不要混用这两层编号。
- 必须为 8 个分布式 rank；启动后核验 `torch_npu`、rank 0～7 ↔ host 后 8 张设备的双射及 `npu-smi` 进程。
- 必须真实调用 MX QR，不允许 CPU、CUDA、`torch.linalg.qr` fallback 或 192×192 bypass。
- Iter 1～30 必须连续、无缺失、无重复且 loss 全部 finite。
- 每一步相对 GPU oracle 的偏差必须 `<=2%`：

  `abs(npu_loss - gpu_loss) / abs(gpu_loss) <= 0.02`

- missing、duplicate、NaN、Inf 或任一步超过 2% 均为失败。

正式比较器：`.codex-tools/step340_loss_gate.py`  
SHA256：`b4e20111333f066183c5474d931b6248129065f4b80cfc9ce7177df5e44d9b7d`

GPU oracle：`.codex-tools/gpu_loss_800.json`  
SHA256：`67b36f3dbb36ff50b2a2bf68062d2e1589e2f55cb94207505fdd504e380a8851`

### 2.2 性能

当前低扰动候选轮只报告 Iter2～30 和稳定窗口耗时，不可单独给最终性能 PASS。

最终性能验收必须用同一环境、同一输入和 fresh process 的 original MX 与 fixed MX paired A/B。CPU fallback、官方 torch QR、旧日志或 GPU 训练耗时只能作外部参照，不能作 fixed/original 分母。

项目方案中暂用最大 10% 回退作为建议值，但用户尚未明确确认。确认前可以采集数据，不能给最终性能 PASS，也不能看完结果后调整阈值。

## 3. 已确认事实、推测与未验证项

### 3.1 已确认事实

- 历史 standalone 原 MX QR 可以通过 finite/重构检查，但端到端 MX 路径曾在 Iter6 出现 NaN。
- STEP326 的特定官方 `torch.linalg.qr` 30-step 合同消除了 NaN，但只有 11/30 loss 满足 2%，且耗时明显偏高。另一个 STEP274 的 192 bypass/no-broadcast 合同曾达到 30/30，但它不是纯 MX 路径。两者都不能替代 MX QrV2 的最终修复。
- STEP392 delta2-only 候选在后 8 卡、world8、真实 192×192 FP32 standalone 输入上，8/8 rank 的 candidate identity、finite、重构、正交、R 下三角及 CPU QR 投影全部通过。
- 上述 STEP392 只能证明候选在孤立路径中没有明显数学错误，不能证明端到端问题已修复。
- STEP393 attempt1～4 都在训练前被夹具问题阻止。
- attempt5 在 runner 静态配置 SHA 门禁停止，未进入 rank ready。
- attempt6 的 host ready waiter 持续到保护窗结束，但 bootstrap child 实际已 rc1，`ready=0/8`，仍未获得训练 step、loss 或 QR 结果。

### 3.2 attempt6 的精确失败边界

- static/base 与 SOAP AST 已通过，MX QR 调用行仍为 `[429, 529]`。
- `environment_preflight.json` 已创建但为 0 字节。
- `host_launcher.log` 的末尾异常类型连续为 `ImportError`，包含 `torch_npu` 标记，不包含 `Config.fromfile` 标记。
- 因此失败发生在 `torch/torch_npu` 导入链，尚未执行 `Config.fromfile`，也没有启动训练 launcher。
- 当前只确认异常阶段和类型；在读取脱敏的精确 ImportError 最终行前，不推测缺少哪个 symbol 或模块。

### 3.3 尚未验证

- delta2-only 候选能否消除真实训练中的 NaN。
- 训练现场是否确实执行该 candidate concrete kernel；当前低扰动轮不启用 profiler，因此不能声称 concrete kernel identity。
- 早先提出的 T/V FreeTensor 后再次读取是否为真实运行时根因。该结论来自源码路径推演，尚无训练现场内部指令/片上内存抓取证明，不能作为最终根因。
- 修复后的 30-step loss 与性能是否过门禁。

## 4. 候选修复与工件身份

当前候选名：`QrV2_qa_position_delta2_only_diagnostic_v1`

候选源码 SHA256：`e352ac31f97980bc5c92caac663414782b78fc3004ff36709b6be8000353b003`

候选是诊断工件，不是 release candidate：

- `diagnostic_only=true`
- `release_candidate=false`
- 禁止打包、安装或覆盖客户环境中的原算子

STEP392 锁定输入与候选构建证据：`STEP392_attempt5_evidence.json`  
SHA256：`90926484a28fbfe7e1e69f52d5154fe2edcfeed683ad9c0e02b06ca7c9ea3fc9`

## 5. STEP393 尝试记录

| Attempt | 到达阶段 | 结果 | 是否产生 E2E 证据 |
|---|---|---|---|
| attempt1 | 远端 source archive 首个 shell 条件 | `[ ! -e path]` 缺空格，训练前失败 | 否 |
| attempt2 | 本地 archive 预检 | 对含 quoted Python 的完整脚本错误执行 `split(';')` | 否 |
| attempt3 | 首次 owned-PGID 观察 | 严格 parser 在确认目标 PGID 前因无关 PID2 的 invalid identity 阻断扫描；`pgrp<=1` 是源码与复现支持的推测，未抓远端 PID2 原始 stat | 否 |
| attempt4 | pre snapshot 加载 guard | backend 内嵌旧 guard SHA，跨文件摘要未传播 | 否 |
| attempt5 | runner 最早静态合同 | STEP193/204 config 已从锁定 SHA 漂移 | 否 |
| attempt6 | environment preflight，rank ready 前 | torch/torch_npu 导入链 ImportError；host 迟至 `0/8 ready timeout` 才退出 | 否 |

每个旧 attempt 目录都必须保留且禁止复用。下一次只能使用新的排他目录，且必须在定位 attempt6 后另行 phase-transition 审核。

## 6. 已修复的夹具问题

### 6.1 STEP393 process guard

文件：`.codex-tools/step393_process_guard.py`

问题：旧预筛在确认 PID 是否属于目标 PGID 前，先调用 STEP377 严格身份 parser。无关内核线程的合法 `pgrp=0/1` 会阻断整个目标组扫描。

修正：

- 新增宽松 stat 预筛；
- 先比较 raw PGID；
- 非目标 PID 直接忽略；
- 只有目标 PGID 才要求有效 starttime，并继续 fail-closed；
- 未修改锁定的 STEP377；
- 未对 PID2 写特判；
- 未放宽目标进程信号授权。

当前 SHA256：`65a15e832d742f3cca2171126ba11e933599632e531e3b41ccdfbf5ffe2c95c0`

聚焦测试：`.codex-tools/test_step393_process_guard.py`，3/3 PASS。

### 6.2 backend/guard 摘要闭包

attempt4 暴露 controller 已更新 guard SHA、backend 仍嵌入旧 SHA 的问题。现已增加 backend 与 controller guard 锁的交叉断言。

backend：`.codex-tools/step393_remote_backend.py`  
当前 SHA256：`9b2fb36842725afe0fe9fd07a3aa12c5f90435ad0f81b380f89e5a13ba94bc98`

### 6.3 canonical config 隔离恢复

历史锁定 config：

- expected size：145464
- expected SHA256：`02aca0c7a33e05e972e268aecc1932bbf10f611fa523ec4696b94d58cd7f56a5`

现场原合同文件已漂移为：

- actual size：145465
- actual SHA256：`79c0146a6d1cb65775d8b98931baceaa391ea13897fbab21a6864726eddb8bf7`

已从项目内完整基线确定性恢复新文件：

`.codex-tools/step393_canonical_aligned_gpu_contract_npu_runtime.py`

恢复后 size/SHA 与历史锁定值完全一致。原合同文件没有被覆盖。

runner 现在：

- 使用 attempt 自带 canonical config；
- 继续只读使用原合同目录中的 launcher 和 training entry；
- 校验 canonical config SHA；
- 校验其绝对 `_base_` 为 regular non-symlink；
- 校验 active base 与锁定 commit 归档中的同相对路径 SHA 相同；
- 在启动 rank 前执行 `Config.fromfile` 解析。

runner：`.codex-tools/run_step393_delta2_shadow_30.sh`  
SHA256：`d5e82a674b4089a4bd506b5e4c583f8adabe7e33987b18f3373e7b94f7c343a4`

## 7. 当前代码状态

controller：`.codex-tools/step393_run_delta2_shadow_30.py`

- 当前 SHA256：`3a94ad60efe65434606fbacfa71dfe856e686e73abd154424f56ce5cd06205c8`
- `E2E_READY=False`
- `PROCESS_GUARD_READY=False`
- 当前目录名仍指向已经用过的 attempt6，禁止再次 arm 或复用。

focused contract test：`.codex-tools/test_step393_delta2_shadow_30.py`

- 1/1 PASS
- 当前 SHA256：`1a8894c13d2252942deec22d5519b750ba85cc9dcaa95694af2aa1afdc48d65c`

最近一次本地验证：

- process guard：3/3 PASS
- controller contract：1/1 PASS
- Python `py_compile`：PASS
- runner `bash -n`：PASS
- `git diff --check`：PASS
- canonical integration 独立审核：P0=0，P1=0

注意：当前 Git 状态同时显示若干文件为 index 中删除、工作树中 untracked。不要执行 `git reset --hard`、`git checkout --`、批量 add 或直接 commit。先查清 index/worktree 来源并只提交本任务明确拥有的新文件。

## 8. attempt6 当前待闭合点

已确认：

- host 最终报 `TimeoutError: STEP393 ready timeout: 0/8`；
- bootstrap child 实际 returncode=1；host waiter 没有及时消费该结果，直到 ready timeout 才退出；
- SOAP AST evidence 完整；
- environment preflight evidence 为 0 字节；
- `host_launcher.log` 证明是涉及 `torch_npu` 的 `ImportError`，发生在 `Config.fromfile` 之前；
- 不能据此归因于 MX QR、NPU 初始化或 HCCL；
- 没有 rank ready，因此 MX QR 尚未执行。

清场已经闭合：`cleanup_postflight` 与 `host_finish.gate` 完整；连续两次 exact case=0、owned PGID=0、端口34393空闲、后8卡NPU进程=0。

接手后的第一步不是重跑。只在远端原位读取并脱敏记录 `host_launcher.log` 的精确 ImportError 最终行和最小调用链，判断是导入顺序、环境变量还是模块内部错误。随后修正两个问题：

1. environment preflight 不要在完成前直接创建最终 JSON；改为临时文件成功后原子发布，失败时单独发布小型结构化 failure JSON。
2. host waiter 在轮询 ready 时同时检查 `bootstrap_result`；child 已 rc1 时立即返回精确阶段错误，不要等完整 ready timeout。

只做一次启动链验证，看到 environment preflight 完整且 8/8 ready 后再运行 30-step。禁止 profiler、tensor dump 或 QR 单算子重放。

## 9. 下一步最小执行顺序

1. 原位读取并脱敏记录 attempt6 的精确 ImportError 最终行和最小调用链。
2. 修复 environment preflight 的原子 success/failure evidence，并让 host waiter 同步消费 bootstrap rc1。
3. 只改 STEP393 新文件并补一个聚焦负例；不要改客户环境或算子。
4. Python 修改后执行代码审查，P0/P1 清零。
5. controller 保持双门禁默认 False；改为全新 attempt 名称。
6. 做一次 phase-transition 审核，并以锁定 controller SHA 的短生命周期进程只在内存 arm。
7. 先做一次最小启动链验证；只有 environment preflight 完整且 8/8 ready，才进入正式 30-step。
8. 只有真实 30-step 完成后才运行 loss gate 和耗时报告。
9. 若低扰动 30-step 正确性通过，再做一次不参与性能裁决的最小训练内 identity 证明，闭合实际执行的是 candidate concrete kernel 而非旧 kernel/fallback。
10. identity 闭合后，按预先锁定阈值执行 fresh original/fixed paired A/B，完成最终性能验收。
11. 若端到端首次出现 nonfinite 或 loss 超 2%，再对确定 step/call 启用最小训练内抓取；不要提前 dump 全量 QR。

## 10. 禁止事项

- 禁止用 standalone PASS 宣称问题已修复。
- 禁止复用 attempt1～6 目录。
- 禁止修改或覆盖漂移的原合同 config。
- 禁止改客户驱动、固件、CANN、PyTorch、torch_npu 或项目依赖版本。
- 禁止在宿主机、CPU、GPU/CUDA 或其他容器运行正式训练。
- 禁止使用 host 前 8 张卡；host 分配/可见集只能为设备8～15，进程内rank 0～7必须与这8张host设备一一映射。
- 禁止把远端日志、tensor、profile、模型、checkpoint 或数据拉到本地。
- 禁止在低扰动性能轮开启 profile/capture/dump、逐 QR `.item()` 或 synchronize。
- 禁止看到结果后放宽 loss 或性能门禁。
- 禁止根据未抓取的片上内存状态把源码推演写成已证实根因。

## 11. 工作区、代码与产物地址

### 11.1 本地工作区

本地项目根目录：

`/home/l30002999/project/MapQr/win-project-backup/DongFeng`

所有本文相对路径都以该目录为基准。不要在其他目录寻找或修改本任务文件。

本地算子包：

`cann-driving-cloud-ops-26.0.7-a3-cann8.3.rc1-torch2.7.1-py3.11-aarch64.zip`

当前对 ZIP 的文件清单检查没有发现 `qr_v2.cpp`/QrV2 C++ 源码条目。`.codex-tools/qr_v2.cpp` 是本项目中的分析工作副本，不是该 ZIP 内已验证的相对路径；对外说明源码位置时必须写清这一区别，不要再把 `.codex-tools/qr_v2.cpp` 冒充成 ZIP 包内路径。

### 11.2 远端运行工作区

以下路径只用于说明交接位置，不包含连接凭据：

- 业务 source repo：`/mnt/sfs_turbo/workdir/wfc1_leicheng/l2.9-df-for-yuexiang`
- 诊断根目录：`/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics`
- 历史 STEP193/204 合同目录：`/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/gpu_contract_alignment_f922c38_8npu_20260814T172611`
- delta2-only 构建工件目录：`/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step385_attempt5_qrv2_delta2_only_opc_build_20260822`
- 运行容器：`mapqr-leicheng`
- STEP393 attempt6：`/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step393_attempt6_delta2_shadow_e2e30_back8_world8_20260822`

attempt1～5 位于同一 diagnostics 根目录，目录名前缀分别为 `step393_attempt1_` 到 `step393_attempt5_`。这些目录和 attempt6 均是只读历史证据，禁止复用、覆盖或删除。

### 11.3 当前核心代码

| 相对路径 | 作用 | 当前状态 |
|---|---|---|
| `.codex-tools/step393_run_delta2_shadow_30.py` | 顶层计划、双门禁、文件SHA闭包、结果校验与远端事务编排 | 默认双门禁False；当前名字指向已用attempt6，禁止直接运行 |
| `.codex-tools/step393_remote_backend.py` | 两跳远端执行、排他目录、上传、source archive、shadow准备、rank/进程/清场控制 | 已接入guard SHA闭包和canonical argv |
| `.codex-tools/run_step393_delta2_shadow_30.sh` | 容器内静态合同、环境预检、8-rank launcher、loss与耗时输出 | 当前在torch/torch_npu导入阶段失败 |
| `.codex-tools/step393_training_entry.py` | 每rank建立最小NPU上下文、发布ready、等待host start gate，再进入原训练入口 | attempt6尚未执行到此文件 |
| `.codex-tools/step393_process_guard.py` | STEP393进程枚举、ownership和安全清场适配器 | 已修无关PID预筛问题 |
| `.codex-tools/step393_canonical_aligned_gpu_contract_npu_runtime.py` | 从项目证据确定性恢复的canonical训练config | 新文件；SHA精确为`02aca0...`，不覆盖原合同 |
| `.codex-tools/test_step393_delta2_shadow_30.py` | STEP393 focused contract test | 1/1 PASS |
| `.codex-tools/test_step393_process_guard.py` | process guard聚焦测试 | 3/3 PASS |
| `.codex-tools/step340_loss_gate.py` | 正式逐步loss门禁 | 规则要求Iter1～30完整且每步<=2%；当前候选尚无E2E结果 |

### 11.4 算子诊断与候选生成代码

| 相对路径 | 用途 |
|---|---|
| `.codex-tools/qr_v2.cpp` | QrV2分析工作副本；来源路径不能表述为ZIP内源码路径 |
| `.codex-tools/step338_patch_qr_v2_lifetime.py` | T/V生命周期方向的诊断patch |
| `.codex-tools/step350_patch_qr_v2_context_capture.py` | QrV2上下文抓取patch |
| `.codex-tools/step372_patch_qr_v2_matmul_position_v5.py` | Matmul position v5候选生成 |
| `.codex-tools/step375_patch_qr_v2_vtv_direct_qa_legacy_probe_v6.py` | V/T/qa单变量probe |
| `.codex-tools/step384_patch_qr_v2_delta2_only_diagnostic.py` | 当前delta2-only诊断候选生成 |
| `.codex-tools/qr_v2_process_patch.cpp` | 过程诊断patch工作文件 |

这些 patch 文件记录的是诊断演进，不代表所有候选都可发布。当前只允许使用已锁定的 delta2-only diagnostic identity。

## 12. 关键文档与证据索引

- `操作步骤.md`：完整时间线、指令、观察和错误尝试。
- `findings.md`：已确认事实与证据边界。
- `progress.md`：阶段进度。
- `task_plan.md`：当前计划状态。
- `MX_QrV2_完整修复实施方案.md`：修复与 release gate 方案。
- `STEP392_attempt5_evidence.json`：delta2-only standalone 工件与输入闭包。

## 13. 面向接手人的一句话

当前最大风险不是“候选 QR 数学错误”，而是尚未让锁定的候选在不改变训练时序的条件下真正跑进 8-rank 端到端路径。先关闭 attempt6 的 torch/torch_npu ImportError 和错误可观测性缺口，再做一次有效 30-step；在此之前，任何“MX QrV2 已修复”或“T/V 生命周期就是最终根因”的结论都超出了证据。
