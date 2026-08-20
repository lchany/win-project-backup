# 东风 MapQR 昇腾 NPU 性能优化适配文档

| 项 | 值 |
|----|-----|
| Git 分支 | `ascend_npu_optimize` |
| 当前 HEAD（QR 修复） | `3a1d763` |
| HTML 对照版 | [`东风MapQR_NPU性能优化适配文档.html`](东风MapQR_NPU性能优化适配文档.html) |
| 正式 config | `projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py` |

## 目录

- [1. 文档说明](#1-文档说明)
- [2. 优化清单](#2-优化清单)
- [3. 分项适配说明](#3-分项适配说明)
- [4. 附录：终稿文件索引](#4-附录终稿文件索引)
- [5. 启用与回退](#5-启用与回退)
- [6. mx QR 试验链说明](#6-mx-qr-试验链说明669a138-之后)

---

## 1. 文档说明

本文整理 MapQR / MV2DFusion（SOAP）在昇腾 NPU 上已落地的**性能适配**与**关键正确性修复**。
每个提交包含：适配对象、原理、动机、落地动作；Diff 以补丁文件为准，分项内附预览。

| 类型 | 说明 |
|------|------|
| **Diff** | `_adapt_doc_materials/diff_<sha>.patch` |
| **Source 终稿** | HTML 版附录含整文件；本地镜像见 `projects/`、`mmcv/`、`tools/` |

**生产启用**：`tools/ddp_train.sh` 默认 `SOAP_STALE_Q_K=4`、`expandable_segments`；配置 `pin_memory=True`。

---

## 2. 优化清单

共 **15** 项（按时间顺序）。涉及文件详见 [§3 分项说明](#3-分项适配说明)。

| # | 提交 | 主题 |
|--:|------|------|
| 1 | [`fb979b2`](#item-fb979b2) | SOAP 预条件器 NPU 亲和优化 |
| 2 | [`5a37d04`](#item-5a37d04) | 训练输入哈希调试移除 |
| 3 | [`14d4f23`](#item-14d4f23) | SOAP 分块 Foreach 调度 |
| 4 | [`b36821e`](#item-b36821e) | GeometricLoss 有限值索引消除 |
| 5 | [`bf9ed6e`](#item-bf9ed6e) | TextLogger 显存统计同步降频 |
| 6 | [`f922c38`](#item-f922c38) | MSDA 切换 DrivingSDK 融合实现 |
| 7 | [`2846401`](#item-2846401) | SOAP 周期 QR 异步流水（stale-Q，k=4） |
| 8 | [`2a2aa0f`](#item-2a2aa0f) | DataContainer 补齐 pin_memory |
| 9 | [`fa95a2a`](#item-fa95a2a) | 训练入口默认 expandable_segments |
| 10 | [`669a138`](#item-669a138) | 配置最终版本修改（不涉及性能修改） |
| 11 | [`10f897d`](#item-10f897d) | mx QR 192×192 绕开 QrV2（暂撤 stale-Q） |
| 12 | [`5899e94`](#item-5899e94) | 撤回误入库 mx_driving_cloud 包 |
| 13 | [`9565044`](#item-9565044) | SOAP 使用 mx_driving_cloud.linalg.qr |
| 14 | [`27b1d6d`](#item-27b1d6d) | 去除随机性固定（对齐 GPU 训练语义） |
| 15 | [`3a1d763`](#item-3a1d763) | SOAP QR 修复：torch.linalg.qr 消除 Iter6 NaN |

---

## 3. 分项适配说明

<a id="item-fb979b2"></a>

### 1. `fb979b2` — SOAP 预条件器 NPU 亲和优化

- **适配对象**：将 SOAP 预条件器路径从偏 CPU / NumPy 亲和实现，调整为在 NPU 上可高效执行的张量路径，并配合训练配置启用相关开关。
- **原理**：SOAP 周期性更新预条件矩阵（含 QR 等线性代数），若大量落在 Host/CPU，会在固定 frequency 的周期步制造秒级空洞。把可亲和部分留在 NPU，减少 D2H/H2D 与 Host 算力占用。
- **为何可优化**：Profiler 显示周期步长尾与 SOAP precondition 强相关；设备侧算子路径可缩短 Host 空闲等待，直接抬升端到端 step time。
- **做了什么**：改写 soap.py 中预条件器相关实现，减少 CPU fallback；配置侧补齐 SOAP 相关启用项。
- **涉及文件**：
  - `projects/mmdet3d_plugin/optimizers/soap.py`
  - `正式 config（mv2dfusion…finetune.py）`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_fb979b2.patch`](_adapt_doc_materials/diff_fb979b2.patch)（261 行）  
> 以下仅展示前 40 行预览。

```diff
diff --git a/projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py b/projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
index 1306a56..05f58f3 100644
--- a/projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
+++ b/projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
@@ -3220,10 +3220,11 @@ optimizer = dict(
         custom_keys={
             "img_backbone": dict(lr_mult=0.25), # 0.25 only for Focal-PETR with R50-in1k pretrained weights
         }
     ),
     precondition_frequency=10,
+    one_sided_dim_threshold=1024,
     weight_decay=0.01)
 
 # optimizer = dict(
 #     type='FASTSOAP', 
 #     lr=max_lr,
diff --git a/projects/mmdet3d_plugin/optimizers/soap.py b/projects/mmdet3d_plugin/optimizers/soap.py
index cfc3987..58347d9 100644
--- a/projects/mmdet3d_plugin/optimizers/soap.py
+++ b/projects/mmdet3d_plugin/optimizers/soap.py
@@ -51,10 +51,11 @@ class SOAP(Optimizer):
         shampoo_beta: float= -1,
         eps: float = 1e-6,
         weight_decay: float = 0.01,
         precondition_frequency: int=10,
         max_precond_dim: int=10000, # 
+        one_sided_dim_threshold: int | None = None,
         merge_dims: bool = False, # Merge dimensions till the product of the dimensions is less than or equal to max_precond_dim.
         precondition_1d: bool = False,
         normalize_grads: bool = False,
         data_format: str = "channels_first",
         correct_bias: bool = True,
@@ -65,10 +66,11 @@ class SOAP(Optimizer):
             "shampoo_beta": shampoo_beta,
             "eps": eps,
             "weight_decay": weight_decay,
             "precondition_frequency": precondition_frequency,
             "max_precond_dim": max_precond_dim,
+            "one_sided_dim_threshold": one_sided_dim_threshold,
             "merge_dims": merge_dims,
```

[↑ 回到清单](#2-优化清单)

---

<a id="item-5a37d04"></a>

### 2. `5a37d04` — 训练输入哈希调试移除

- **适配对象**：删除训练前向中仅为 loss 对齐/调试服务的输入哈希与相关同步打印路径。
- **原理**：调试哈希往往触发额外 Host 同步、字符串/哈希计算甚至设备到主机的标量搬运，对训练语义无贡献却拉长每步。
- **为何可优化**：对齐阶段遗留的调试代码在正式训练中仍执行，属于明确可删的性能噪音。
- **做了什么**：从 spetr3d 检测器中移除输入哈希调试逻辑。
- **涉及文件**：
  - `projects/mmdet3d_plugin/models/detectors/spetr3d.py`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_5a37d04.patch`](_adapt_doc_materials/diff_5a37d04.patch)（30 行）

```diff
diff --git a/projects/mmdet3d_plugin/models/detectors/spetr3d.py b/projects/mmdet3d_plugin/models/detectors/spetr3d.py
index 820c1be..9962658 100644
--- a/projects/mmdet3d_plugin/models/detectors/spetr3d.py
+++ b/projects/mmdet3d_plugin/models/detectors/spetr3d.py
@@ -488,25 +488,11 @@ class SPetr3D(MVXTwoStageDetector):
                             data_tag=None,
                             pred_bbox3d_range_valid=None,
                             depth_map=None,
                             **data):
         losses = dict()
-#随机性固定
-        print("FWD_IN",
-            "img", tensor_hash(data["img"]),
-            "points", tensor_hash(data["points"]) if "points" in data else None,
-            "gt_bboxes_3d", tensor_hash(gt_bboxes_3d),
-            "map_gt_bboxes_3d", tensor_hash(map_gt_bboxes_3d))
-#随机性固定
         T = data['img'].size(1)     # torch.Size([2, 1, 7, 3, 544, 960])
-#随机性固定
-        print("FWD_IN",
-            "img", tensor_hash(data["img"]),
-            "points", tensor_hash(data["points"]) if "points" in data else None,
-            "gt_bboxes_3d", tensor_hash(gt_bboxes_3d),
-            "map_gt_bboxes_3d", tensor_hash(map_gt_bboxes_3d))
-#随机性固定
         num_nograd_frames = T - self.num_frame_head_grads
         num_grad_losses = T - self.num_frame_losses # 0
         if gt_motions_3d is None:
             gt_motions_3d = [None] * T
         if gt_bboxes_3d_cam is None:
```

[↑ 回到清单](#2-优化清单)

---

<a id="item-14d4f23"></a>

### 3. `14d4f23` — SOAP 分块 Foreach 调度

- **适配对象**：对 SOAP 中大量同构小更新，改为分块 / foreach 风格批量调度，降低 Python 逐参数开销与细碎 kernel launch。
- **原理**：优化器逐步对每个参数做同类 op 时，Python 循环 + 多次小 kernel 会放大 Host 调度成本；foreach / 分块可合并同类更新。
- **为何可优化**：SOAP 参数量大，逐步更新的调度开销在 NPU 上更易暴露；合并后减少 launch 与解释器开销。
- **做了什么**：重构 soap.py 中相关更新路径为分块 foreach 调度，保持数值语义门禁通过后提交。
- **涉及文件**：
  - `projects/mmdet3d_plugin/optimizers/soap.py`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_14d4f23.patch`](_adapt_doc_materials/diff_14d4f23.patch)（249 行）  
> 以下仅展示前 40 行预览。

```diff
diff --git a/projects/mmdet3d_plugin/optimizers/soap.py b/projects/mmdet3d_plugin/optimizers/soap.py
index 58347d9..413d9cb 100644
--- a/projects/mmdet3d_plugin/optimizers/soap.py
+++ b/projects/mmdet3d_plugin/optimizers/soap.py
@@ -104,114 +104,140 @@ class SOAP(Optimizer):
             new_shape.append(curr_shape)
         
         new_grad = grad.reshape(new_shape)
         return new_grad     
     
-    @torch.no_grad()
-    def step(self):
-        """
-        Performs a single optimization step.
-
-        Arguments:
-            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
-        """
-        loss = None
-
-        for group in self.param_groups:
-            for p in group["params"]:
-                if p.grad is None:
-                    continue
-                p.grad.nan_to_num_()
-                grad = p.grad
-
-                state = self.state[p]
-                
-                if "step" not in state:
-                    state["step"] = 0 
-                    
-                # State initialization
-                if "exp_avg" not in state:
-                    # Exponential moving average of gradient values
-                    state["exp_avg"] = torch.zeros_like(grad)
-                    # Exponential moving average of squared gradient values
-                    state["exp_avg_sq"] = torch.zeros_like(grad)
-
-                if 'Q' not in state:
```

[↑ 回到清单](#2-优化清单)

---

<a id="item-b36821e"></a>

### 4. `b36821e` — GeometricLoss 有限值索引消除

- **适配对象**：去掉 GeometricLoss 中依赖有限值掩码再索引的路径，改为更亲和、更少同步的实现。
- **原理**：`isfinite` + `nonzero`/`index` 一类模式常引入 AICPU / Host 同步；在可证明等价时用掩码算术或向量化路径替代。
- **为何可优化**：该路径在 profile 中有可回收耗时，且可做输出/梯度门禁验证。
- **做了什么**：改写 geo_loss.py 有限值索引逻辑，正式短训门禁通过后独立提交。
- **涉及文件**：
  - `projects/mmdet3d_plugin/models/losses/geo_loss.py`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_b36821e.patch`](_adapt_doc_materials/diff_b36821e.patch)（83 行）  
> 以下仅展示前 40 行预览。

```diff
diff --git a/projects/mmdet3d_plugin/models/losses/geo_loss.py b/projects/mmdet3d_plugin/models/losses/geo_loss.py
index 042cbcb..0d0b4f8 100644
--- a/projects/mmdet3d_plugin/models/losses/geo_loss.py
+++ b/projects/mmdet3d_plugin/models/losses/geo_loss.py
@@ -70,10 +70,32 @@ def _compute_inter_geometrics(
     # dots = dots.masked_select(mask)
     # cross = cross.masked_select(mask)
 
     return length.reshape(-1), dots.reshape(-1), cross.reshape(-1)
 
+
+def _finite_l1_loss(input, target, size_average=None, reduction='mean'):
+    """L1 reduction over finite targets without materializing index tensors."""
+    if size_average is not None or reduction not in ('mean', 'sum'):
+        finite = torch.isfinite(target)
+        return l1_loss(input[finite], target[finite], size_average,
+                       reduction=reduction)
+
+    finite = torch.isfinite(target)
+    difference = torch.where(
+        finite, torch.abs(input - target), torch.zeros_like(input))
+    total = difference.sum()
+    if reduction == 'sum':
+        return total
+
+    count = finite.sum()
+    finite_mean = total / count.clamp_min(1)
+    # Match l1_loss(empty, empty, reduction='mean'): NaN value with a
+    # zero gradient. The branch stays on-device and does not synchronize.
+    empty_mean = total * 0.0 + input.new_tensor(float('nan'))
+    return torch.where(count > 0, finite_mean, empty_mean)
+
 @LOSSES.register_module()
 class GeometricLoss(nn.Module):
     """
         Implementation of Geometric Loss
     Args:
@@ -191,16 +213,16 @@ class GeometricLoss(nn.Module):
             ft_targets = normalized_target[valid_mask]
             # ft_targets_denormalized = target[..., :2][valid_mask]
```

[↑ 回到清单](#2-优化清单)

---

<a id="item-bf9ed6e"></a>

### 5. `bf9ed6e` — TextLogger 显存统计同步降频

- **适配对象**：降低 TextLogger 中触发设备同步的显存统计频率，避免每 step 为打日志而 synchronize。
- **原理**：读取 NPU 显存或某些 memory 字段常隐含 synchronize，若每 iteration 执行，会把异步计算强制拉齐，拉长 step。
- **为何可优化**：日志精度不需要逐步实时显存；降频后日志仍可用，同步成本大幅下降。
- **做了什么**：TextLogger 改为按间隔统计显存；配置侧配合调整。
- **涉及文件**：
  - `mmcv/runner/hooks/logger/text.py`
  - `正式 config（mv2dfusion…finetune.py）`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_bf9ed6e.patch`](_adapt_doc_materials/diff_bf9ed6e.patch)（75 行）

```diff
diff --git a/mmcv/runner/hooks/logger/text.py b/mmcv/runner/hooks/logger/text.py
index 3f75900..6ab7a20 100644
--- a/mmcv/runner/hooks/logger/text.py
+++ b/mmcv/runner/hooks/logger/text.py
@@ -32,10 +32,13 @@ class TextLoggerHook(LoggerHook):
         reset_flag (bool, optional): Whether to clear the output buffer after
             logging. Default: False.
         interval_exp_name (int, optional): Logging interval for experiment
             name. This feature is to help users conveniently get the experiment
             information from screen or log file. Default: 1000.
+        memory_interval (int, optional): Interval for synchronizing the maximum
+            allocated memory across distributed ranks. The latest sampled value
+            is reused between synchronizations. Default: 1.
         out_dir (str, optional): Logs are saved in ``runner.work_dir`` default.
             If ``out_dir`` is specified, logs will be copied to a new directory
             which is the concatenation of ``out_dir`` and the last level
             directory of ``runner.work_dir``. Default: None.
             `New in version 1.3.16.`
@@ -57,18 +60,23 @@ class TextLoggerHook(LoggerHook):
                  by_epoch: bool = True,
                  interval: int = 10,
                  ignore_last: bool = True,
                  reset_flag: bool = False,
                  interval_exp_name: int = 1000,
+                 memory_interval: int = 1,
                  out_dir: Optional[str] = None,
                  out_suffix: Union[str, tuple] = ('.log.json', '.log', '.py'),
                  keep_local: bool = True,
                  file_client_args: Optional[Dict] = None):
         super().__init__(interval, ignore_last, reset_flag, by_epoch)
         self.by_epoch = by_epoch
         self.time_sec_tot = 0
         self.interval_exp_name = interval_exp_name
+        if memory_interval < 1:
+            raise ValueError('memory_interval must be a positive integer')
+        self.memory_interval = memory_interval
+        self._max_memory_mb = None
 
         if out_dir is None and file_client_args is not None:
             raise ValueError(
                 'file_client_args should be "None" when `out_dir` is not'
                 'specified.')
@@ -230,11 +238,14 @@ class TextLoggerHook(LoggerHook):
                 log_dict['lr'].update({k: lr_[0]})
 
         if 'time' in runner.log_buffer.output:
             # statistic memory
             if torch.cuda.is_available():
-                log_dict['memory'] = self._get_max_memory(runner)
+                if (self._max_memory_mb is None
+                        or cur_iter % self.memory_interval == 0):
+                    self._max_memory_mb = self._get_max_memory(runner)
+                log_dict['memory'] = self._max_memory_mb
 
         log_dict = dict(log_dict, **runner.log_buffer.output)  # type: ignore
 
         self._log_info(log_dict, runner)
         self._dump_log(log_dict, runner)
diff --git a/projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py b/projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
index c2319bf..b0d06d8 100644
--- a/projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
+++ b/projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
@@ -50,11 +50,11 @@ custom_imports = dict(
     allow_failed_imports=False
 )
 log_config = dict(
     interval=1,
     hooks=[
-        dict(type='TextLoggerHook'),
+        dict(type='TextLoggerHook', memory_interval=10),
         dict(type='TensorboardLoggerHook')
     ])
 
 # custom_hooks = [
 #     dict(type='UnusedParamCheckHook'),   
```

[↑ 回到清单](#2-优化清单)

---

<a id="item-f922c38"></a>

### 6. `f922c38` — MSDA 切换 DrivingSDK 融合实现

- **适配对象**：Multi-Scale Deformable Attention 前向/反向从通用/次优实现切换为 DrivingSDK 融合算子路径。
- **原理**：MSDA 是 BEV/Map 结构中的高频重算子；融合 kernel 减少中间临时张量、格式转换与多次 launch，提高 AI Core 利用率。
- **为何可优化**：普通步 profile 中 MSDA forward/backward 长期位居 TopN；SDK 融合实现在固定环境内可落地且通过函数级/长训门禁。
- **做了什么**：修改 multi_scale_deformable_attn_function.py 的调用入口，走 DrivingSDK；876-step、checkpoint/resume、推理对照通过后采用。
- **涉及文件**：
  - `projects/mmdet3d_plugin/models/utils/multi_scale_deformable_attn_function.py`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_f922c38.patch`](_adapt_doc_materials/diff_f922c38.patch)（69 行）

```diff
diff --git a/projects/mmdet3d_plugin/models/utils/multi_scale_deformable_attn_function.py b/projects/mmdet3d_plugin/models/utils/multi_scale_deformable_attn_function.py
index 514c3d1..42322f9 100644
--- a/projects/mmdet3d_plugin/models/utils/multi_scale_deformable_attn_function.py
+++ b/projects/mmdet3d_plugin/models/utils/multi_scale_deformable_attn_function.py
@@ -3,10 +3,11 @@
 # ---------------------------------------------
 #  Modified by Zhiqi Li
 # ---------------------------------------------
 
 import torch
+import mx_driving
 from torch.cuda.amp import custom_bwd, custom_fwd
 from torch.autograd.function import Function, once_differentiable
 from mmcv.utils import ext_loader
 ext_module = ext_loader.load_ext(
     '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
@@ -113,17 +114,17 @@ class MultiScaleDeformableAttnFunction_fp32(Function):
         Returns:
             Tensor: has shape (bs, num_queries, embed_dims)
         """
 
         ctx.im2col_step = im2col_step
-        output = ext_module.ms_deform_attn_forward(
-            value,
-            value_spatial_shapes,
-            value_level_start_index,
-            sampling_locations,
-            attention_weights,
-            im2col_step=ctx.im2col_step)
+        value_spatial_shapes = value_spatial_shapes.int()
+        value_level_start_index = value_level_start_index.int()
+        sampling_locations = sampling_locations.type_as(value)
+        attention_weights = attention_weights.type_as(value)
+        output = mx_driving._C.multi_scale_deformable_attn(
+            value, value_spatial_shapes, value_level_start_index,
+            sampling_locations, attention_weights)
         ctx.save_for_backward(value, value_spatial_shapes,
                               value_level_start_index, sampling_locations,
                               attention_weights)
         return output
         
@@ -174,23 +175,12 @@ class MultiScaleDeformableAttnFunction_fp32(Function):
              Tuple[Tensor]: Gradient
                 of input tensors in forward.
         """
         value, value_spatial_shapes, value_level_start_index, \
             sampling_locations, attention_weights = ctx.saved_tensors
-        grad_value = torch.zeros_like(value)
-        grad_sampling_loc = torch.zeros_like(sampling_locations)
-        grad_attn_weight = torch.zeros_like(attention_weights)
-
-        ext_module.ms_deform_attn_backward(
-            value,
-            value_spatial_shapes,
-            value_level_start_index,
-            sampling_locations,
-            attention_weights,
-            grad_output.contiguous(),
-            grad_value,
-            grad_sampling_loc,
-            grad_attn_weight,
-            im2col_step=ctx.im2col_step)
+        grad_value, grad_sampling_loc, grad_attn_weight = \
+            mx_driving._C.multi_scale_deformable_attn_backward(
+                value, value_spatial_shapes, value_level_start_index,
+                sampling_locations, attention_weights, grad_output)
 
         return grad_value, None, None, \
             grad_sampling_loc, grad_attn_weight, None
```

[↑ 回到清单](#2-优化清单)

---

<a id="item-2846401"></a>

### 7. `2846401` — SOAP 周期 QR 异步流水（stale-Q，k=4）

- **适配对象**：对 SOAP 周期 QR：在侧流异步计算新 Q，主路径继续使用陈旧 Q，固定 k=4 步后换入；默认 k=0 保持同步语义。
- **原理**：QR 在周期步极重。数学上投影用的 Q 更新频率本就低于逐步；允许有限步陈旧 Q，可把 QR 与后续普通步重叠，摊销周期长尾。
- **为何可优化**：固定环境无更快且状态逐位等价的 QR primitive；在不改 QR 公式本身的前提下，异步+陈旧换入是唯一通过长门禁的大幅优化。
- **做了什么**：实现侧流 QR、pending 安装与环境变量 SOAP_STALE_Q_K（默认 0，生产启用 4）；Stage A–D 与正式训练门禁通过。
- **涉及文件**：
  - `projects/mmdet3d_plugin/optimizers/soap.py`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_2846401.patch`](_adapt_doc_materials/diff_2846401.patch)（219 行）  
> 以下仅展示前 40 行预览。

```diff
diff --git a/projects/mmdet3d_plugin/optimizers/soap.py b/projects/mmdet3d_plugin/optimizers/soap.py
index 413d9cb..fbdc6b5 100644
--- a/projects/mmdet3d_plugin/optimizers/soap.py
+++ b/projects/mmdet3d_plugin/optimizers/soap.py
@@ -1,5 +1,6 @@
+import os
 import torch
 import torch.nn as nn
 from torch.optim import Optimizer
 from mmcv.runner.optimizer import OPTIMIZERS
 from itertools import chain
@@ -122,10 +123,12 @@ class SOAP(Optimizer):
         params = [item[0] for item in chunk]
         grads = [item[1] for item in chunk]
         states = [item[2] for item in chunk]
         group = chunk[0][3]
         beta1, beta2 = group["betas"]
+        for state in states:
+            self._stale_q_install_if_due(state)
         grad_projected = [
             self.project(grad, state, merge_dims=group["merge_dims"],
                          max_precond_dim=group["max_precond_dim"])
             for grad, state in zip(grads, states)
         ]
@@ -332,11 +335,14 @@ class SOAP(Optimizer):
                      
         if state['Q'] is None:
             state['Q'] = self.get_identity_basis(state['GG'])
             state['Q'] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)
         if state['step'] > 0 and state['step'] % state['precondition_frequency'] == 0:
-            state['Q'] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)           
+            if self._stale_q_eligible(state, merge_dims):
+                self._stale_q_submit(state, max_precond_dim, merge_dims)
+            else:
+                state['Q'] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)           
 
     def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
         """
         Projects the gradient back to the original space.
         """
```

[↑ 回到清单](#2-优化清单)

---

<a id="item-2a2aa0f"></a>

### 8. `2a2aa0f` — DataContainer 补齐 pin_memory

- **适配对象**：为 MMCV DataContainer 实现真正的 pin_memory()，使配置里 pin_memory=True 生效。
- **原理**：页锁定内存可加速 Host→Device 的 DMA 拷贝。若 DataContainer 未实现 pin，DataLoader 的 pin 对包装后的 tensor 无效。
- **为何可优化**：在 stale-Q 降低 SOAP 周期成本后，数据搬运占比上升；补齐 pin 后 100-step 正式 A/B 吞吐显著提升。
- **做了什么**：新增 DataContainer.pin_memory，递归 pin 内部 tensor；与配置 train_loader.pin_memory=True 配合。
- **涉及文件**：
  - `mmcv/parallel/data_container.py`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_2a2aa0f.patch`](_adapt_doc_materials/diff_2a2aa0f.patch)（28 行）

```diff
diff --git a/mmcv/parallel/data_container.py b/mmcv/parallel/data_container.py
index 62f2573..d79783a 100644
--- a/mmcv/parallel/data_container.py
+++ b/mmcv/parallel/data_container.py
@@ -87,5 +87,23 @@ class DataContainer:
         return self.data.size(*args, **kwargs)
 
     @assert_tensor_type
     def dim(self) -> int:
         return self.data.dim()
+
+    def pin_memory(self):
+        """Pin CPU tensor payload so DataLoader pin_memory=True takes effect."""
+        if self.cpu_only:
+            return self
+
+        def _pin(obj):
+            if isinstance(obj, torch.Tensor):
+                return obj.pin_memory()
+            if isinstance(obj, list):
+                return [_pin(x) for x in obj]
+            if isinstance(obj, tuple):
+                return tuple(_pin(x) for x in obj)
+            return obj
+
+        self._data = _pin(self._data)
+        return self
+
```

[↑ 回到清单](#2-优化清单)

---

<a id="item-fa95a2a"></a>

### 9. `fa95a2a` — 训练入口默认 expandable_segments

- **适配对象**：在 canonical 训练入口默认设置 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True。
- **原理**：NPU caching allocator 的 expandable segments 减少显存碎片与不必要的扩缩，可轻微降低峰值并改善分配抖动。
- **为何可优化**：正式跑已长期带着该设置；单变量 A/B 显示吞吐小幅正向且 peak 下降，适合作为启动脚本卫生项固化。
- **做了什么**：在 tools/ddp_train.sh 增加一行 export；不改模型代码。
- **涉及文件**：
  - `tools/ddp_train.sh`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_fa95a2a.patch`](_adapt_doc_materials/diff_fa95a2a.patch)（14 行）

```diff
diff --git a/tools/ddp_train.sh b/tools/ddp_train.sh
index 26aaf4c..17226b7 100644
--- a/tools/ddp_train.sh
+++ b/tools/ddp_train.sh
@@ -1,8 +1,9 @@
 \#!/usr/bin/env bash
 
 set -x
+export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
 export FUNCTION=${1}
 
 DEBUG=${DEBUG:-false}
 MODE=${MODE:-single}
 
```

[↑ 回到清单](#2-优化清单)

---

<a id="item-669a138"></a>

### 10. `669a138` — 配置最终版本修改（不涉及性能修改）

- **适配对象**：把昨日 876 步 GPU 合同配置与原名训练入口固化为仓库唯一正式版本，去掉诊断 overlay / 双配置。
- **原理**：性能算子（SOAP stale-Q、pin_memory、expandable_segments）已在此前提交落地；本提交只对齐正式 config 与 tools/ddp_train.sh
- **为何可优化**：不改算子公式。目的是交付可直接 bash tools/ddp_train.sh tools/train_spetr.py <正式config>
- **做了什么**：正式 config：use_grid_mask=True
- **涉及文件**：
  - `正式 config（mv2dfusion…finetune.py）`
  - `tools/ddp_train.sh`
  - `tools/train_spetr.py`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_669a138.patch`](_adapt_doc_materials/diff_669a138.patch)（209 行）  
> 以下仅展示前 40 行预览。

```diff
diff --git a/projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py b/projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
index b0d06d8..d245fc6 100644
--- a/projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
+++ b/projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py
@@ -52,7 +52,7 @@ custom_imports = dict(
 log_config = dict(
     interval=1,
     hooks=[
-        dict(type='TextLoggerHook', memory_interval=10),
+        dict(type='TextLoggerHook', memory_interval=10),
         dict(type='TensorboardLoggerHook')
     ])
 
@@ -332,7 +332,7 @@ tasks = [
 #num_gpus = 8
 #随机性固定
 #随机性固定
-num_gpus = 8
+num_gpus = 8
 #随机性固定
 import torch
 # if ('4090' in torch.cuda.get_device_name()) or ('L4' in torch.cuda.get_device_name()) :
@@ -343,7 +343,7 @@ import torch
 # batch_size = 16
 #随机性固定
 #随机性固定
-batch_size = 16
+batch_size = 16
 #随机性固定
 num_iters_per_epoch = 28130 // (num_gpus * batch_size)
 num_epochs = 4
@@ -374,7 +374,7 @@ model = dict(
     num_frame_backbone_grads=num_frame_losses,
     num_frame_losses=num_frame_losses,
     #随机性固定
-    use_grid_mask=False,
+    use_grid_mask=True,
     #随机性固定
     fix_backbone=fix_backbone,
     fix_pts_backbone = fix_pts_backbone,
```

[↑ 回到清单](#2-优化清单)

---

<a id="item-10f897d"></a>

### 11. `10f897d` — mx QR 192×192 绕开 QrV2（暂撤 stale-Q）

- **适配对象**：在 SOAP 中接入 mx_driving_cloud.linalg.qr，并临时入库本地 linalg 包装（192×192 走 torch QR 绕开 QrV2）；同期撤掉 stale-Q 异步路径以便单变量验证 mx QR。
- **原理**：QrV2 在部分设备/shape 上存在末 tile 与设备绑定缺陷；192×192 固定走 AICPU torch QR 可规避 BAD dump，其余 shape 仍走 mx 算子以追求 NPU 性能。
- **为何可优化**：Profiler 显示 SOAP 周期 QR 是最大瓶颈；mx QR 若稳定可用可显著缩短周期步。
- **做了什么**：新增 mx_driving_cloud/ops/linalg.py（48 行）；soap.py 改 mx QR 并移除 stale-Q 相关方法与周期分支。
- **涉及文件**：
  - `mx_driving_cloud/ops/linalg.py`
  - `projects/mmdet3d_plugin/optimizers/soap.py`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_10f897d.patch`](_adapt_doc_materials/diff_10f897d.patch)（1147 行）  
> 以下仅展示前 40 行预览。

```diff
diff --git a/mx_driving_cloud/__init__.py b/mx_driving_cloud/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/mx_driving_cloud/ops/__init__.py b/mx_driving_cloud/ops/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/mx_driving_cloud/ops/linalg.py b/mx_driving_cloud/ops/linalg.py
new file mode 100644
index 0000000..5788c3a
--- /dev/null
+++ b/mx_driving_cloud/ops/linalg.py
@@ -0,0 +1,48 @@
+# Copyright (c) 2026 Huawei Technologies Co., Ltd.
+# NPU QR: 192x192 uses torch.linalg.qr; larger matrices use QrV2 when device path is healthy.
+
+import torch
+import torch.nn.functional as F
+from torch.autograd import Function
+
+import mx_driving_cloud._C
+
+BLOCK_TILING = 64
+QR_AICPU_THRESHOLD_SHAPE = 80
+QR_SOAP_FIXED_SHAPE = 192
+
+
+class QR(Function):
+    @staticmethod
+    def forward(ctx, A: torch.Tensor):
+        dim = A.shape
+        if len(dim) != 2:
+            raise ValueError(
+                f"Input tensor must be 2D, got {len(dim)}D shape {dim}")
+
+        if dim[0] <= QR_AICPU_THRESHOLD_SHAPE or dim[1] <= QR_AICPU_THRESHOLD_SHAPE:
+            return torch.linalg.qr(A)
+
+        if dim[0] == QR_SOAP_FIXED_SHAPE and dim[1] == QR_SOAP_FIXED_SHAPE:
+            return torch.linalg.qr(A)
+
```

[↑ 回到清单](#2-优化清单)

---

<a id="item-5899e94"></a>

### 12. `5899e94` — 撤回误入库 mx_driving_cloud 包

- **适配对象**：删除误提交进业务仓库的 mx_driving_cloud 源码树，改回仅使用客户环境 driving-cloud-ops wheel。
- **原理**：算子包应由客户 CANN/wheel 统一交付；仓库内嵌副本会与 site-packages 版本漂移并造成双路径维护。
- **为何可优化**：10f897d 的本地 linalg 仅为临时 bypass 试验，不应作为正式交付物留在 Git 中。
- **做了什么**：删除 mx_driving_cloud/ 三个占位/包装文件（48 行 linalg 逻辑撤回）；SOAP 仍保留 mx QR 调用，运行时走 wheel。
- **涉及文件**：
  - `mx_driving_cloud/__init__.py`
  - `mx_driving_cloud/ops/__init__.py`
  - `mx_driving_cloud/ops/linalg.py`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_5899e94.patch`](_adapt_doc_materials/diff_5899e94.patch)（60 行）

```diff
diff --git a/mx_driving_cloud/__init__.py b/mx_driving_cloud/__init__.py
deleted file mode 100644
index e69de29..0000000
diff --git a/mx_driving_cloud/ops/__init__.py b/mx_driving_cloud/ops/__init__.py
deleted file mode 100644
index e69de29..0000000
diff --git a/mx_driving_cloud/ops/linalg.py b/mx_driving_cloud/ops/linalg.py
deleted file mode 100644
index 5788c3a..0000000
--- a/mx_driving_cloud/ops/linalg.py
+++ /dev/null
@@ -1,48 +0,0 @@
-# Copyright (c) 2026 Huawei Technologies Co., Ltd.
-# NPU QR: 192x192 uses torch.linalg.qr; larger matrices use QrV2 when device path is healthy.
-
-import torch
-import torch.nn.functional as F
-from torch.autograd import Function
-
-import mx_driving_cloud._C
-
-BLOCK_TILING = 64
-QR_AICPU_THRESHOLD_SHAPE = 80
-QR_SOAP_FIXED_SHAPE = 192
-
-
-class QR(Function):
-    @staticmethod
-    def forward(ctx, A: torch.Tensor):
-        dim = A.shape
-        if len(dim) != 2:
-            raise ValueError(
-                f"Input tensor must be 2D, got {len(dim)}D shape {dim}")
-
-        if dim[0] <= QR_AICPU_THRESHOLD_SHAPE or dim[1] <= QR_AICPU_THRESHOLD_SHAPE:
-            return torch.linalg.qr(A)
-
-        if dim[0] == QR_SOAP_FIXED_SHAPE and dim[1] == QR_SOAP_FIXED_SHAPE:
-            return torch.linalg.qr(A)
-
-        lda = max(dim[0], dim[1])
-        if lda == 0:
-            return (A, A)
-        if dim[0] == 1:
-            return (torch.ones(1, 1, dtype=A.dtype, device=A.device), A)
-
-        pad = lda % BLOCK_TILING
-        pad = BLOCK_TILING - pad if pad else 0
-        lda_pad = lda + pad
-        A_pad = F.pad(
-            A,
-            (0, lda_pad - dim[1], 0, lda_pad - dim[0]),
-        ).contiguous()
-        Q, R = mx_driving_cloud._C.qr(A_pad)
-        Q = Q[: dim[0], : dim[0]]
-        R = R[: dim[0], : dim[1]]
-        return Q, torch.triu(R)
-
-
-qr = QR.apply
```

[↑ 回到清单](#2-优化清单)

---

<a id="item-9565044"></a>

### 13. `9565044` — SOAP 使用 mx_driving_cloud.linalg.qr

- **适配对象**：在恢复 stale-Q（k=4）异步流水的前提下，将两处 QR 调用切换为客户 mx_driving_cloud.linalg.qr。
- **原理**：stale-Q 把 AICPU QR 移出关键路径；mx QR 若在 NPU 上更快，可进一步压缩周期步墙时。
- **为何可优化**：STEP-284 目标是把 torch.linalg.qr 替换为 NPU 原生 QR，与 fb979b2 + 2846401 性能栈叠加。
- **做了什么**：soap.py：import mx_driving_cloud；get_orthogonal_matrix_QR / _qr_finish 改用 mx_driving_cloud.linalg.qr；恢复 STEP-221 stale-Q 全套方法与周期分支。
- **涉及文件**：
  - `projects/mmdet3d_plugin/optimizers/soap.py`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_9565044.patch`](_adapt_doc_materials/diff_9565044.patch)（1088 行）  
> 以下仅展示前 40 行预览。

```diff
diff --git a/projects/mmdet3d_plugin/optimizers/soap.py b/projects/mmdet3d_plugin/optimizers/soap.py
index 59bcd2d..77e412c 100644
--- a/projects/mmdet3d_plugin/optimizers/soap.py
+++ b/projects/mmdet3d_plugin/optimizers/soap.py
@@ -1,121 +1,201 @@
-import torch
-import torch.nn as nn
-from torch.optim import Optimizer
-from mmcv.runner.optimizer import OPTIMIZERS
-from itertools import chain
-import os
-import mx_driving_cloud
+import os
+import torch
+import mx_driving_cloud
+import torch.nn as nn
+from torch.optim import Optimizer
+from mmcv.runner.optimizer import OPTIMIZERS
+from itertools import chain
+
+# Parts of the code are modifications of Pytorch's AdamW optimizer
+# Parts of the code are modifications of code from https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector.py
+@OPTIMIZERS.register_module()
+class SOAP(Optimizer):
+    """
+    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321).
+
+    Parameters:
+        params (`Iterable[nn.parameter.Parameter]`):
+            Iterable of parameters to optimize or dictionaries defining parameter groups.
+        lr (`float`, *optional*, defaults to 0.003):
+            The learning rate to use.
+        betas (`Tuple[float,float]`, *optional*, defaults to `(0.95, 0.95)`):
+            Adam's betas parameters (b1, b2).
+        shampoo_beta (`float`, *optional*, defaults to -1):
+            If >= 0, use this beta for the preconditioner (L and R in paper, state['GG'] below) moving average instead of betas[1].
+        eps (`float`, *optional*, defaults to 1e-08):
+            Adam's epsilon for numerical stability.
+        weight_decay (`float`, *optional*, defaults to 0.01): weight decay coefficient.
+        precondition_frequency (`int`, *optional*, defaults to 10):
```

[↑ 回到清单](#2-优化清单)

---

<a id="item-27b1d6d"></a>

### 14. `27b1d6d` — 去除随机性固定（对齐 GPU 训练语义）

- **适配对象**：移除为 loss 对齐临时加入的随机性固定、msprobe 与额外 hook，恢复与客户 GPU 训练语义一致的代码路径。
- **原理**：随机性固定与额外 debugger 会改变数据顺序、算子选择与 Host 侧开销，不属于性能优化交付物。
- **为何可优化**：669a138 之后仍需在正式 config/数据管线上撤掉诊断期随机性约束，避免与 GPU 基线合同偏离。
- **做了什么**：删除 optimizer hook 中的固定逻辑；config 去掉临时字段；dataset/spetr3d 移除 msprobe/随机性注释块与相关 import。
- **涉及文件**：
  - `mmcv/runner/hooks/optimizer.py`
  - `正式 config`
  - `core/hook`
  - `internal_dataset_track_stream`
  - `vectorize_local_map`
  - `spetr3d.py`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_27b1d6d.patch`](_adapt_doc_materials/diff_27b1d6d.patch)（566 行）  
> 以下仅展示前 40 行预览。

```diff
diff --git a/mmcv/runner/hooks/optimizer.py b/mmcv/runner/hooks/optimizer.py
index ccbd9f7..269285a 100644
--- a/mmcv/runner/hooks/optimizer.py
+++ b/mmcv/runner/hooks/optimizer.py
@@ -55,59 +55,6 @@ def _model_grad_hash(model):
         h.update(_tensor_hash(p.grad).encode())
     return h.hexdigest()[:16]
 
-#随机性固定
-
-def _debug_hash(value):
-    if value is None:
-        return "NONE"
-    data = value.detach().cpu().contiguous().numpy().tobytes()
-    return hashlib.sha256(data).hexdigest()[:16]
-
-
-def _print_grad_hashes(model, cur_iter, prefixes=("lane3d_head", "pts_bbox_head")):
-    m = model.module if hasattr(model, "module") else model
-    for name, p in m.named_parameters():
-        if not any(prefix in name for prefix in prefixes):
-            continue
-        if p.grad is None:
-            continue
-        print("GRAD_PARAM", cur_iter, name, _tensor_hash(p.grad))
-        
-target = (
-    "lane3d_head.transformer.decoder.layers.0"
-    ".attentions.1.sampling_offsets.weight"
-)
-
-def _sync():
-    if IS_NPU_AVAILABLE:
-        torch.npu.synchronize()
-
-def _optimizer_param_hashes(runner):
-    model = runner.model.module if hasattr(runner.model, "module") else runner.model
-    names = {id(p): n for n, p in model.named_parameters()}
-
-    result = []
```

[↑ 回到清单](#2-优化清单)

---

<a id="item-3a1d763"></a>

### 15. `3a1d763` — SOAP QR 修复：torch.linalg.qr 消除 Iter6 NaN

- **适配对象**：STEP-324~326 验证 mx QR 与 SOAP 下游（sort_idx / exp_avg_sq / 预条件状态）Q/R 约定不兼容导致 Iter6+ NaN；正式修复为 torch.linalg.qr（SOAP 原设计语义）。
- **原理**：mx 与 torch 的 Q/R 分解非等价（非单纯符号差）；SOAP 依赖 torch 约定做特征排序与状态更新，mx 路径 Iter6 起 seg loss NaN。
- **修复动机**：这是功能正确性修复，不是性能回退：torch QR 30 步全 finite，SOAP 周期步仍 ~5 s（STEP-326 已验证无 steady-state 耗时劣化）。
- **做了什么**：删除 import mx_driving_cloud；两处 QR 调用改回 torch.linalg.qr；不保留 SOAP_QR_BACKEND 运行时兼容层。
- **涉及文件**：
  - `projects/mmdet3d_plugin/optimizers/soap.py`

#### Diff

> **完整 Diff**：[`_adapt_doc_materials/diff_3a1d763.patch`](_adapt_doc_materials/diff_3a1d763.patch)（29 行）

```diff
diff --git a/projects/mmdet3d_plugin/optimizers/soap.py b/projects/mmdet3d_plugin/optimizers/soap.py
index 77e412c..fbdc6b5 100644
--- a/projects/mmdet3d_plugin/optimizers/soap.py
+++ b/projects/mmdet3d_plugin/optimizers/soap.py
@@ -1,6 +1,5 @@
 import os
 import torch
-import mx_driving_cloud
 import torch.nn as nn
 from torch.optim import Optimizer
 from mmcv.runner.optimizer import OPTIMIZERS
@@ -426,7 +425,7 @@ class SOAP(Optimizer):
             exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
             o = o.index_select(1, sort_idx)
             power_iter = m @ o
-            Q, _ = mx_driving_cloud.linalg.qr(power_iter)
+            Q, _ = torch.linalg.qr(power_iter)
             if Q.dtype != original_dtype:
                 Q = Q.to(original_dtype)
             final.append(Q)
@@ -526,7 +525,7 @@ class SOAP(Optimizer):
             if entry is None:
                 result.append(None)
                 continue
-            Q, _ = mx_driving_cloud.linalg.qr(entry["power_iter"])
+            Q, _ = torch.linalg.qr(entry["power_iter"])
             if Q.dtype != entry["original_dtype"]:
                 Q = Q.to(entry["original_dtype"])
             result.append(Q)
```

[↑ 回到清单](#2-优化清单)

---

## 4. 附录：终稿文件索引

| 路径 | 说明 |
|------|------|
| `projects/configs/20260113st/mv2dfusion_ppheavy_r34_0114_v2.0.4_sep-range_far3d_relu6_no-lidar-ca_new-bb_gop_occ_finetune.py` | 正式训练 config |
| `tools/ddp_train.sh` | 训练入口（`SOAP_STALE_Q_K=4`、`expandable_segments`） |
| `tools/train_spetr.py` | 训练脚本 |
| `projects/mmdet3d_plugin/optimizers/soap.py` | SOAP 优化器（HEAD `3a1d763`） |
| `projects/mmdet3d_plugin/models/utils/multi_scale_deformable_attn_function.py` | MSDA DrivingSDK |
| `projects/mmdet3d_plugin/models/losses/geo_loss.py` | GeometricLoss |
| `projects/mmdet3d_plugin/models/detectors/spetr3d.py` | 检测器 |
| `mmcv/parallel/data_container.py` | `pin_memory` |
| `mmcv/runner/hooks/logger/text.py` | TextLogger |

---

## 5. 启用与回退

| 场景 | 操作 |
|------|------|
| 完整收益栈 | `tools/ddp_train.sh` 默认 `SOAP_STALE_Q_K=4` + `expandable_segments`，配置 `pin_memory=True` |
| stale-Q 回退 | `ddp_train.sh` 中 `SOAP_STALE_Q_K=4` → `0` |
| allocator 回退 | 去掉 `ddp_train.sh` 中 `expandable_segments` 行 |
| 配置/入口回退 | revert `669a138` |
| SOAP QR（当前正式） | `torch.linalg.qr`（`3a1d763`，NaN 根因修复，非性能回退） |
| 随机性固定 | 若需恢复诊断期行为，revert `27b1d6d` |

---

## 6. mx QR 试验链说明（669a138 之后）

```text
669a138 → 10f897d → 5899e94 → 9565044 → 27b1d6d → 3a1d763 (HEAD)
```

| 顺序 | 提交 | 性质 |
|--:|------|------|
| 1 | `10f897d` | mx QR 试验 + 暂撤 stale-Q |
| 2 | `5899e94` | 撤回误入库 `mx_driving_cloud` 包 |
| 3 | `9565044` | 恢复 stale-Q + mx QR（引入 NaN 风险） |
| 4 | `27b1d6d` | 去除随机性固定（对齐 GPU 语义，非性能项） |
| 5 | `3a1d763` | **NaN 根因修复**（torch QR，非性能回退） |

**STEP-324~326 结论**：mx QR 与 SOAP 下游 Q/R 约定不兼容 → Iter6+ NaN；torch QR 30/30 finite，steady SOAP 步 ~5 s。

