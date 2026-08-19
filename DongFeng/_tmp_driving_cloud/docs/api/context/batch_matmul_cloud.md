## npu_batch_matmul
### 接口原型
```python
mx_driving_cloud.batch_matmul_cloud(Tensor projection_mat, Tensor pts_extend) -> Tensor
```

### 功能描述
矩阵相乘。
### 参数说明
- `projection_mat(Tensor)`：投影矩阵，数据类型为`bf16, fp16, fp32`，支持维度1-6维，且和pts_extend互相可广播，满足矩阵相乘关系。
- `pts_extend(Tensor)`：所有点的特征，数据类型为`bf16, fp16, fp32`，支持维度1-6维，且和projection_mat互相可广播，满足矩阵相乘关系。
### 返回值
- `output(Tensor)`：矩阵乘结果，数据类型为`bf16, fp16, fp32`。
### 支持的型号
- Atlas A2 训练系列产品
### 优化说明
当满足如下Shape且数据类型为`float32`，调用 mx_driving.npu_batch_matmul，其余情况调用 torch.matmul
- **projection_mat**：Shape为4-6维，最后两维需要是`4, 4`或`3, 3`，且和pts_extend互相可广播，满足矩阵相乘关系。
- **pts_extend**：Shape为4-6维，最后两维需要是`4, 1`或`3, 1`，且和projection_mat互相可广播，满足矩阵相乘关系。
### 调用示例
输入是6维
```python
import numpy as np
import torch, torch_npu
import mx_driving_cloud
torch.npu.config.allow_internal_format = True

projection_mat = torch.randn((6, 6, 4, 4)).npu()
pts_extend = torch.randn(6, 1220, 13, 4).npu()
projection_mat_fused = projection_mat[:, :, None, None].contiguous()
pts_extend2_fused = pts_extend[:, None, ..., None].contiguous()
projection_mat_fused.requires_grad = True
pts_extend2_fused.requires_grad = True
result = mx_driving_cloud.batch_matmul_cloud(projection_mat_fused, pts_extend2_fused)
grad = torch.ones_like(result)
result.backward(grad)
```
输入是4维
```python
import numpy as np
import torch, torch_npu
import mx_driving_cloud
torch.npu.config.allow_internal_format = True

projection_mat = torch.randn((6, 1220, 4, 4)).npu()
pts_extend = torch.randn(6, 1220, 4, 1).npu()
projection_mat.requires_grad = True
pts_extend.requires_grad = True
result = mx_driving_cloud.batch_matmul_cloud(projection_mat, pts_extend)
grad = torch.ones_like(result)
result.backward(grad)
```