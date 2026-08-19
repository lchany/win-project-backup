## futr3d_target_single
### 接口原型
```python
mx_driving_cloud.futr3d_target_single(
    pos_inds: torch.Tensor,
    pos_assigned_gt_inds: torch.Tensor,
    pos_gt_bboxes: torch.Tensor,
    bbox_pred: torch.Tensor,
    gt_bboxes: torch.Tensor,
    gt_labels: torch.Tensor,
    num_classes: int,
    code_size: int,
    dist_loss_weight: bool,
    has_train_label_weights: bool,
    train_label_weights: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]
```
### 功能描述
对futr3d_dino_pred_head.py中_get_target_single方法进行算子融合 
### 参数说明
- `pos_inds(Tensor)`: 正样本索引，数据类型支持`int64`
- `pos_assigned_gt_inds(Tensor)`: 正样本分配的真实目标索引，数据类型支持`int64`
- `pos_gt_bboxes(Tensor)`: 正样本对应的真实边界框，数据类型支持`float32`
- `bbox_pred(Tensor)`: 模型预测的边界框，数据类型支持`float32`
- `gt_bboxes(Tensor)`: 所有真实目标的边界框，数据类型支持`float32`
- `gt_labels(Tensor)`: 所有真实目标的类别标签，数据类型支持`int64`
- `num_classes`: 总类别数，数据类型支持`int64`
- `code_size`: 边界框的参数维度，数据类型支持`int64`
- `dist_loss_weight`: 是否使用距离相关的损失权重，数据类型支持`bool`
- `has_train_label_weights`: 是否存在训练标签的权重，数据类型支持`bool`
- `train_label_weights(Tensor)`: 训练标签的权重，可选，数据类型支持`float32`

### 返回值
- `labels(Tensor)`: 样本的分类标签，数据类型为`int64`
- `label_weights(Tensor)`: 分类标签权重，数据类型为`float32`
- `bbox_targets(Tensor)`: 边界框回归的目标值，数据类型为`float32`
- `bbox_weights(Tensor)`: 边界框回归的权重，数据类型为`float32`

### 约束说明


### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving_cloud import futr3d_target_single

bbox_pred = torch.rand((600, 10), dtype=torch.float32).npu()
gt_bboxes = torch.rand((29, 9), dtype=torch.float32).npu()
gt_labels = torch.randint(0, 2, (29,), dtype=torch.int64).npu()
pos_inds = torch.randperm(599)[:29].npu()
pos_assigned_gt_inds = torch.randint(0, 28, (29,), dtype=torch.int64).npu()
pos_gt_bboxes = torch.rand((29, 9), dtype=torch.float32).npu()
num_classes = 13
code_size = 10
dist_loss_weight = True
has_train_label_weights = True
train_label_weights = torch.rand((600,), dtype=torch.float32).npu()
out = futr3d_target_single(
    pos_inds,
    pos_assigned_gt_inds,
    pos_gt_bboxes,
    bbox_pred,
    gt_bboxes,
    gt_labels,
    num_classes,
    code_size,
    dist_loss_weight,
    has_train_label_weights,
    train_label_weights,
)
```