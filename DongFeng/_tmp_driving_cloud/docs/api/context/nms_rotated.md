## nms_rotated
### 接口原型
```python
mx_driving_cloud.nms_rotated(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor keep
```
### 功能描述
对旋转框进行非极大值抑制计算，boxes根据scores排序，iou大于iou_threshold的低score的box将被抑制
### 参数说明
- `boxes(Tensor)`：框张量，数据类型为`float32`。shape 为`[N, 5]`。`5`分别代表`x, y, w, h, angle`。
- `scores(Tensor)`：评分张量，数据类型为`float32`。shape 为`[N]`。
- `iou_threshold(float)`：IoU阈值。
### 返回值
- `keep(Tensor)`：被保留的box下标，数据类型为`int16`。
### 约束说明
- boxes的shape应为(N，5)
- scores的shape应为(N)
- N <= 15000
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving_cloud import nms_rotated
boxes = torch.tensor([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]], dtype=torch.float32).npu()
scores = torch.tensor([1, 2], dtype=torch.float32).npu()
keep = nms_rotated(boxes, scores, 0.5)
```