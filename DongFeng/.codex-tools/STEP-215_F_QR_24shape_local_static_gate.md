# STEP-215-F：QR 24-shape 本地静态门禁

## 裁决

`PASS_STATIC_PACKAGE_READY_NOT_EXECUTED`

本报告只证明本地待执行包的语法、来源摘要和控制合同闭合；没有连接远端、没有进入容器、没有启动 NPU，也不代表 24-shape 数值或性能门禁已经通过。

## 来源完整性

- 四个 STEP-215 源文件的 SHA256 已固化在 `step215_qr_24shape_local_source_manifest.sha256`；清单只含文件名和摘要，不含主机路径、远端路径、地址、端口、账号或凭据。
- runner 复用既有 `step214_j_ready_controller.py`。runner 会在正式输出目录另行记录 harness、controller、summarizer 的 SHA256；本地四文件清单补充固化 host launcher 与 runner 自身。
- 四个已固化文件均未因本次审计改写。

## 语法与静态结构

- Codex bundled Git runtime 的 `usr/bin/sh.exe` 实际报告 GNU Bash `5.2.37(1)-release`；以该解释器对 container runner 和 host launcher 执行 `-n`，两个退出码均为 `0`。
- harness、summarizer 以及复用 controller 均通过 Python AST parse；该检查不导入 `torch`/`torch_npu`，因而不会创建设备上下文。
- AST 字面量复算得到 `24` 类 shape、历史 `551` 次；活动合同为去掉 `4×5120`，并把 `2560` 从 `8` 次降为 `4` 次，合计 `543` 次。

## 计时与数值门禁合同

- 每个 shape 在正式采样前分别执行一次 `linalg.qr` 和 `geqrf+orgqr` warmup。
- 样本数分支为 `5/3/2`，所以所有 shape 的 baseline/candidate 均至少有 `2` 个正式样本；顺序按 sample 与 rank 奇偶交替。
- baseline 与 candidate 各记录 Q 的重复自噪声；直接 Q 的 NRMSE 使用自噪声自适应阈值，但硬上限始终为 `1e-5`。
- 硬门禁字段为：Q max-abs `<=1e-5`、Q NRMSE `<=1e-5`、R NRMSE `<=1e-5`、正交 max-abs `<=1e-5`、正交 normalized-Fro `<=1e-5`、重构 relative-L2 `<=1e-5`，并要求 finite 与输入摘要/指针不变。
- summarizer 聚合 8 rank 的逐 shape 最坏数值误差、rank 中位 Event 时延、HBM 增量，以及按当前 `543` 次活动权重计算的周期收益。

## Controller、超时与释放合同

- host launcher 精确匹配唯一 `mapqr-leicheng` 容器，并在启动前要求训练/profiler 活动进程数为 `0`。
- runner 固定 `ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`，使用单机 `8` rank；harness 再校验 world size、rank/local-rank 映射和 visible 字符串。
- controller 要求 `8/8` ready、rank 映射、world8 和 payload gate，再在 rank 仍存活时采集并核验 `npu-smi` 的 8 个物理进程。
- 超时链为 controller `900s`，宿主外层 `930s` 且 TERM 后 `5s` KILL；rank 在 ready 后等待 release 的独立上限为 `180s`。
- controller 在 `finally` 中无条件创建 release 文件并原子写状态；runner 的 `EXIT/INT/TERM` cleanup 也会创建 release、终止并 wait 进程组；宿主对 `124/137` 再补一次 release/TERM。三层释放路径静态闭合。
- `gate_pass` 只表示 rank 已完成并保持 live，数值是否合格由独立的 `numeric_gate_pass` 表达；因此局部数值 FAIL 可以被完整收集，而不会被 controller 误判为控制协议失败。

## 本地验证命令（脱敏表示）

```text
<codex-git-runtime>/usr/bin/sh.exe --version
<codex-git-runtime>/usr/bin/sh.exe -n step215_qr_24shape_run_inside_container.sh
<codex-git-runtime>/usr/bin/sh.exe -n step215_qr_24shape_host_launch_contract.sh
python -  # 仅 AST parse/字面量复算，不导入项目运行时
Get-FileHash -Algorithm SHA256 <四个STEP-215源文件>
```

下一步仍需受控远端通道恢复后，先校验这四个 SHA，再按既定资源门禁运行唯一一次 world8 局部 A/B；本静态 PASS 不授权应用 SOAP patch、训练或提交。
