## refline_head_get_target_single
### 接口原型
```python
mx_driving_cloud.refline_head_get_target_single(bbox_pred: torch.Tensor, 
                                                gt_bboxes: torch.Tensor,
                                                pts_pred: torch.Tensor, 
                                                gt_labels: torch.Tensor,
                                                pos_assigned_gt_inds: torch.Tensor,
                                                pos_inds: torch.Tensor,  
                                                gt_shifts_pts: torch.Tensor,
                                                gt_routes_vecs_valid: torch.Tensor,
                                                pos_gt_bboxes: torch.Tensor,      
                                                route_class_num: int,
                                                order_index: Optional[torch.Tensor])
                                                -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```
### 功能描述
对refline_head的get_target_single方法进行算子融合 
### 参数说明
- `bbox_pred (Tensor)`：数据类型为float32，shape为(bbox_pred_size0, bbox_pred_size1)
- `gt_bboxes (Tensor)`：数据类型为float32，shape为(gt_bboxes0, gt_bboxes1)
- `pts_pred (Tensor)`：数据类型为float32，shape为(pts_pred_size0, pts_pred_size1, pts_pred_size2)
- `gt_labels (Tensor)`：数据类型为int64，shape为(gt_labels_size0,)
- `pos_assigned_gt_inds (Tensor)`：即sampling_result.pos_assigned_gt_inds数据类型为int64，shape为(pos_assigned_gt_inds_size0,)
- `pos_inds (Tensor)`：即sampling_result.pos_inds，数据类型为int64，shape为(pos_inds_size0,)
- `gt_shifts_pts (Tensor)`：数据类型为float32，shape为(gt_shifts_pts_size0, gt_shifts_pts_size1, gt_shifts_pts_size2, gt_shifts_pts_size3)	
- `gt_routes_vecs_valid (Tensor)`：数据类型为float32，shape为(gt_routes_vecs_valid_size0, gt_routes_vecs_valid_size1, gt_routes_vecs_valid_size2)
- `pos_gt_bboxes (Tensor)`：数据类型为float32，shape为(pos_gt_bboxes,)
- `route_class_num (int)`：数据类型为uint32_t
- `order_index (Optional[torch.Tensor])`：数据类型为optional<tensor> 或<tensor>	torch.int64	None，shape为torch.
### 返回值
- `label_weights (Tensor)`：数据类型为float32，shape为(bbox_pred_size0,)
- `labels (Tensor)`：数据类型为int64，shape为(bbox_pred_size0,)
- `bbox_targets (Tensor)`：数据类型为float32，shape为(bbox_pred_size0, gt_bboxes1)
- `bbox_weights (Tensor)`：数据类型为float32，shape为(bbox_pred_size0, bbox_pred_size1)
- `pts_targets (Tensor)`：数据类型为float32，shape为(pts_pred_size0, pts_pred_size1, pts_pred_size2)
- `pts_weights (Tensor)`：数据类型为float32，shape为(pts_pred_size0, pts_pred_size1, pts_pred_size2)
- `pts_targets_valid (Tensor)`：数据类型为float32，shape为(pts_pred_size0, pts_pred_size1, pts_pred_size2)
### 约束说明
- `pos_inds`长度不能超过 UINT32_MAX - 8
- `gt_labels`长度不能超过 UINT32_MAX - 8
- 数据类型为`float32`时，`0 < bbox_pred.size(1) < 16384`
- 数据类型为`float32`时，`0 < pts_pred_size.size(1) * pts_pred_size.size(2) < 16384`
- 数据类型为`float32`时，`0 < gt_bboxes.size(-1) < 16384`
- 数据类型为`int64`时，`order_index.size(-1) < 8129`

### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving_cloud import refline_head_get_target_single

# 生成 [0,1) 随机数 → 缩放至 [-5,5)（公式：rand * 10 - 5）
bbox_pred = (torch.rand((15, 4), dtype=torch.float32) * 10 - 5).npu()  
gt_bboxes = (torch.rand((2, 4), dtype=torch.float32) * 10 - 5).npu()   
pts_pred = (torch.rand((15, 50, 2), dtype=torch.float32) * 10 - 5).npu()  
gt_shifts_pts = (torch.rand((2, 1, 50, 2), dtype=torch.float32) * 10 - 5).npu()  
gt_routes_vecs_valid = (torch.rand((2, 50, 2), dtype=torch.float32) * 10 - 5).npu()  
pos_gt_bboxes = (torch.rand((10, 4), dtype=torch.float32) * 10 - 5).npu() 

gt_labels = torch.randint(0, 2, (2,), dtype=torch.int64).npu() 
pos_assigned_gt_inds = torch.randint(0, 2, (10,), dtype=torch.int64).npu()  
pos_inds = torch.randperm(15)[:10].npu()  
route_class_num = 5
order_index = None

out = refline_head_get_target_single(
    bbox_pred,
    gt_bboxes,
    pts_pred, 
    gt_labels,
    pos_assigned_gt_inds,
    pos_inds,  
    gt_shifts_pts,
    gt_routes_vecs_valid,
    pos_gt_bboxes,
    route_class_num,
    order_index,
)

