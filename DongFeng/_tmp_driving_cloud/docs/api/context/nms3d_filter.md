## nms3d_filter
### 接口原型
```python
mx_driving_cloud.nms3d_filter(Tensor boxes, Tensor scores, float: iou_threshold) -> Tensor
```
### 功能描述
3D非极大值抑制，在bev视角下剔除多个3d box交并比大于阈值的box。
### 参数说明
- `boxes(Tensor)`：框张量，数据类型为`float32`。shape 为`[N, 7]`。`7`分别代表`x, y, z, x_size, y_size, z_size, rz`。
- `scores(Tensor)`：评分张量，数据类型为`float32`。shape 为`[N]`。
- `iou_threshold(float)`：IoU阈值。
### 返回值
- `output(Tensor)`：NMS后的框张量，数据类型为`int16`。
### 约束说明
- boxes的shape应为(N, 7)
- scores的shape应为(N, 1)
- N <= 15000
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving_cloud import nms3d_filter
boxes = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]], dtype=torch.float32).npu()
scores = torch.tensor([1, 2], dtype=torch.float32).npu()
keep = nms3d_filter(boxes, scores, 0.5)
```