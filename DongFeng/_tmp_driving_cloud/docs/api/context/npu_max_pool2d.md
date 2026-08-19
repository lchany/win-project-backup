## npu_max_pool2d
### 接口原型
```python
mx_driving_cloud.npu_max_pool2d(Tensor x, int kernel_size, int stride, int padding) -> Tensor
```
### 功能描述
对输入进行最大池化，并输出最大池化值。
### 参数说明
- `x (Tensor)`：一组待池化对象，数据类型支持`float32`、`float16`，format为NCHW，输入数据量不超过10亿。
### 返回值
- `y (Tensor)`：池化后的最大值，数据类型与`x (Tensor)`相同，format为NCHW。
### 约束说明
1. kernel_size仅支持3，stride支持1、2，padding仅支持1，H和W需要大于1。
2. 数据类型为`float32`时，C不超过1360；数据类型为`float16`时，C不超过2720。
3. 性能在C值较大的场景下较优，建议使用规格为C>=64。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving_cloud import npu_max_pool2d
kernel_size = 3
stride = 2
padding = 1
x = torch.randn(18, 64, 464, 800).npu()
res = npu_max_pool2d(x, kernel_size, stride, padding)
```