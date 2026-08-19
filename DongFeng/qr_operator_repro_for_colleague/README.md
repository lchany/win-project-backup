# mx_driving_cloud QR 问题输入（给算子同事）

> **2026-08-19 STEP-301 纠正：** 旧复现脚本把 A 搬到 `npu:k`，但没有先执行
> `torch.npu.set_device(k)`。因此旧报告中的 visible npu2–7 `507015` 是
> current-device/输入-device 不一致导致的诊断脚本错误，不能再作为 QrV2
> “设备分域失效”的证据。修正脚本后，8 卡可见的 npu1–7 与七张物理卡
> 单独可见均 7/7 有限、无 507015。训练态末 64 列非有限仍是独立待复现问题。

包版本：`mx_driving_cloud 26.0.7`（`cann-driving-cloud-ops-26.0.7-a3-cann8.3.rc1-torch2.7.1-py3.11-aarch64.zip`）  
接口：`mx_driving_cloud.linalg.qr(A)` → `(Q, R)`  
复现时请**关闭**任何 192 bypass（不要设 `MX_QR_VALIDATION_BYPASS`，`ops/linalg.py` 里也不要写死 192→`torch.linalg.qr`）。

## 当前单算子复验结论

单次冷调用通常正常，但仍不能据此判定训练态末 tile 问题已经消失。

我们用同一套 26.0.7、关闭 bypass、CPU FP64 当金标，扫了 SOAP 全部 24 种方阵（88 例独立进程）：

| 测法 | 结果 |
|---|---|
| 单卡 **npu0**，`eye` / `randn` / `1e-8` 小幅值，以及本包 BAD `.pt` | **经常全有限**。今天冷跑 BAD192 的 `Q@R−A` 约 `2e-14` |
| 旧脚本：8 张可见卡，输入放到 npu2，但 current device 留在 npu0 | 曾触发 `507015`；该用例无效，已撤销 |
| 修正脚本：8 张可见卡，先 `set_device(k)` 再在 npu1–7 调用 BAD A | **7/7 有限、无 507015** |
| 训练进程里同一次 SOAP 周期 dump 出的 Q/R | BAD 样本 **末 64 列非有限**（见 `.pt` 里的 `Q`/`R`） |

请不要只用「npu0 + 随机/单位阵 + 调一次」当验收。那种测法我们这边也对得上：没 NaN。

正确的多卡可见调用必须先绑定 current device：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # 或 8-15，只要可见下标能写到 2
python repro_qr.py --eye --npu 2
```

修正后的期望：Q/R 有限，且 `Q@R−A` 很小。若要复现训练态末 64 列问题，需保留正确设备绑定并构造训练内高频/状态化调用，不能再靠 current-device 错配制造 507015。

`.pt` 里的 `Q`/`R` 是**训练现场已经算坏的输出快照**，用来证明「A 合法、Q 非法」。把这份 A 再单独 qr 一次，npu0 上经常又能算对——说明缺陷是 **最后一块 64-tile 的释放后仍 CalcQ（use-after-free）**，不是这份矩阵数值无解。源码：`qr_v2.cpp` 在 `k=blockp-1` 时 `colNum=0`/`useCoreNum=0`，LARFB 释放 `vLocal` 后 core0 仍 `CalcQForLARFB`。

## 发什么文件

`.pt` 用来对照训练态已经坏掉的输出与合法输入；它不是冷调用必现 NaN/崩溃的输入。  
8 个 rank 的 **A 完全相同**。看训练 dump 用 `rank0_step10_ind0_192x192_BAD.pt` 即可。

| 文件 | 大小 | 内容 |
|---|---|---|
| `inputs/rank{0-7}_step10_ind0_192x192_BAD.pt` | 各 445031 B | dict：`A`,`Q`,`R`,`meta` |

加载：

```python
import torch
obj = torch.load("rank0_step10_ind0_192x192_BAD.pt", map_location="cpu", weights_only=False)
A, Q, R = obj["A"], obj["Q"], obj["R"]   # fp32, [192,192]
print(obj["meta"])
```

也可用同目录 `repro_qr.py`。

## 样本从哪来

- 训练：8 卡 SOAP，第一次周期 `opt_step=10`，`factor_ind=0`
- 当时一次周期共 4408 次 QR，其中 **8 次** `[192,192]` 为 BAD（每 rank 1 次）
- 同周期另有 248 次 192×192 正常（对照样本 `192x192_SAMPLE`，`recon_max≈6.4e-8`，未放进本包）
- dtype：`float32`，device：当时各 rank 自己的 NPU

## 输入 A（合法）

8 份 A 统计一致：

- 全有限，无 0 / denormal
- `min≈-7.48e-8`，`max≈7.91e-8`，`absmax≈7.91e-8`
- `cond2(fp64)≈1763`，Frobenius ≈ `1.64e-6`
- numpy CPU FP64 / FP32 `qr(A)` **均成功**（FP64 `recon_max≈1e-22`，FP32 ≈`4e-15`）

结论：不是 SOAP 把 A 喂坏。

## 输出 Q/R（算子坏）

| 张量 | 有限？ | 非有限个数 | 位置 |
|---|---|---|---|
| A | 是 | 0 | — |
| Q | 否 | 12288 = 192×64 | **列 128–191**（最后一个 64-tile） |
| R | 否 | 10272 | 对应最后 64 列 |

`meta.recon_max` 为 null：Q/R 非有限，无法做 `Q@R-A`。

## 两种失败（请都测）

### 1) 训练态 NaN（这份 BAD A）

在 **npu0 / npu1** 上对 `A.npu()` 调 `mx_driving_cloud.linalg.qr`：  
有时能算出有限值，训练 in-process 高频调用更容易得到**最后 64 列非有限**（与本 dump 一致）。

源码对应：`qr_v2.cpp` 最后一块 `k=blockp-1`（192 时 `blockp=3`，k=2），`InitTaskTiling(k+1)` 得到 `colNum=0` / `useCoreNum=0`，LARFB 全核释放 `vLocal` 后 **core0 仍 `CalcQForLARFB`**。

### 2) 已撤销：旧“设备分域 507015”结论

旧 harness 在 `ASCEND_RT_VISIBLE_DEVICES` 一组 8 卡时，直接把输入放到
`npu2–7`，但 current device 仍是默认 npu0，随后出现：

`AclrtSynchronizeDeviceWithTimeout 507015`  
kernel：`QrV2_*_mix_aic`，`MTE instruction DDR address out of range`

≤80 走 AICPU `torch.linalg.qr`，所以 **64×64 不崩**。  
STEP-301 在每个独立进程先执行 `torch.npu.set_device(k)` 并断言
`current_device==k` 后，npu1–7 全部正常。因此该 507015 来自设备上下文错误，
不是物理卡或 logical npu2–7 固有故障。

最小复现（不需要本 `.pt`）：

```python
import os, torch, torch_npu, mx_driving_cloud
torch.npu.set_device(2)
A = torch.eye(192, device="npu:2")   # 在 visible 组里用 npu2–7
Q, R = mx_driving_cloud.linalg.qr(A)
torch.npu.synchronize()
```

## 建议同事怎么跑

1. 容器 CANN / torch / torch_npu 与客户环境一致；安装上述 26.0.7 wheel。  
2. **不要** Python bypass。  
3. 独立进程，并在创建/搬运输入前显式 `torch.npu.set_device(npu)`：  
   - npu0：`A` = 本 dump 的 `obj["A"]`  
   - npu2：`eye(192)` 以及同一份 `A`  
4. 先打印并核对 `torch.npu.current_device()`；再检查 `torch.isfinite(Q/R).all()`；非有限时看 `~isfinite(Q).any(0)` 是否落在列 128–191。  
5. CPU 对照：`torch.linalg.qr(A.cpu())` 应有限。

## 尚未 dump 的现象（本次 800 step）

只把 192 绕开 `QrV2` 后，训练 Iter1–95 loss 有限，**Iter96 起 `loss: nan`**（紧挨 SOAP Iter94）。  
说明 **只修 192 不够**，其它 `>80` 的 shape（128/160/220/256/440/768/1024/2560/5120 等）仍可能坏。那些 SAMPLE `.pt` 在第一次周期是有限的，**不是**本次 nan 的直接证据，故未打进本包。
