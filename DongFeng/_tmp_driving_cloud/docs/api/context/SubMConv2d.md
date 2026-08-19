## SubMConv2d
### 接口原型
```python
mx_driving_cloud.SubMConv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, indice_key=None, mode='mmcv') -> SparseConvTensor
```

### 功能描述
稀疏卷积，只有当卷积核中心参与计算时，才会影响输出
### 参数说明
- `in_channels(int)`：输入数据的通道数
- `out_channels(int)`：输出通道数
- `kernel_size(List(int)/Tuple(int)/int)`：卷积神经网络中卷积核的大小
- `stride(List(int)/Tuple(int)/int)`：卷积核在输入数据上滑动时的步长
- `dilation(List(int)/Tuple(int)/int)`：空洞卷积大小
- `groups(int)`：分组卷积
- `bias(bool)`：偏置项
- `indice_key(str)`：该输入用于复用之前计算的索引信息
- `mode(str)`：区分了`mmcv`和`spconv`两种不同框架下的稀疏卷积
### 返回值
- `SparseConvTensor(Tensor)`：存储了输出的特征值`out_feature`，对应索引位置`out_indices`和对应的spatital_shape。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
- `kernel_size`当前支持数据类型为二维List/Tuple或Int，当前值仅支持1、3
- `stride`当前支持数据类型为二维List/Tuple或Int,当前仅支持值为1
- `dilation`，`groups`当前仅支持值为1
### 调用示例
```python
import torch,torch_npu
import numpy as np
from mx_driving_cloud import SubMConv2d, SparseConvTensor

def generate_indice(batch, height, width, actual_num):
    base_indices = np.random.permutation(np.arange(batch * height * width))[:actual_num]
    base_indices = np.sort(base_indices)
    b_indice = base_indices // (height * width)
    base_indices = base_indices % (height * width)
    h_indice = base_indices // (width)
    w_indice = base_indices % (width)
    indices = np.concatenate((b_indice, h_indice, w_indice)).reshape(3, actual_num)
    return indices

actual_num = 20
batch_size = 4
spatial_shape =  [9, 9]
in_channels = 16
out_channels = 32
kernel_size = 3
stride = 1
padding = 0
dtype = torch.float32
mode = "spconv"
indices = torch.from_numpy(generate_indice(batch_size, spatial_shape[0], spatial_shape[1], actual_num)).int().transpose(0, 1).contiguous().npu()
feature = torch.rand(actual_num, in_channels, dtype=dtype).npu()
x = SparseConvTensor(feature, indices, spatial_shape, batch_size)
net = SubMConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, mode=mode).npu()
weight = torch.rand(net.weight.size(), dtype=dtype)
net.weight.data = weight.npu()
out = net(x) 
```