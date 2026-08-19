# unique_dim

## 产品支持情况

| 产品                                                     | 是否支持 |
| -------------------------------------------------------- | :------: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |

## 函数原型
```python
mx_driving_cloud.unique_dim(input, sorted=True, return_inverse=False, return_counts=False, dim=None) -> tuple[Tensor, Tensor, Tensor]
```

## 功能说明
* API功能：对输入的点云数据进行去重处理。

## 参数说明
- **input**(Tensor)：必选参数，输入数据，数据格式支持ND，支持非连续的Tensor。
- **sorted**(bool)：可选参数，是否对唯一元素按升序排序后输出，默认值为True，数据类型支持bool。
- **return_inverse**(bool)：可选参数，是否还返回原始输入中的元素最终出现在返回的唯一列表中的索引，默认值为False，数据类型支持bool。
- **return_counts**(bool)：可选参数，是否返回每个唯一元素的计数，默认值为False，数据类型支持bool。
- **dim**(int)：可选参数，待操作的维度，默认值为None，数据类型支持int32。如果为`None`，则返回展平输入的唯一值。否则，每个由给定维度索引的张量都将被视为要应用唯一值操作的元素之一。

## 返回值说明
- **output**(​Tensor)：去重后的​Tensor，数据类型与input (Tensor)相同。
- **inverse_indices**(​Tensor​)：索引数据（可选），若return_inverse设为True，会额外返回一个张量（与输入形状相同），用于表示原始输入中的元素在输出中对应的索引，数据类型为`int64`。
- **counts**(​Tensor​)：可选，若return_counts设为True，会额外返回一个张量（若指定了dim，则形状与输出或output.size (dim)相同），用于表示每个唯一值或唯一张量的出现次数，数据类型为`int64`。

## 约束说明
1. 满足以下条件时，使用优化的自定义算子。参数约束参考[参数说明](#参数说明)一节。
- **input**：shape为`[N,3]`，数据类型为`int32`，$N \le 2^{31}-1$，每个元素的值必须大于等于0，三列最大值加1后的乘积不超过32位整数的表示范围。小shape（N<=5500）建议直接使用torch.unique。
- **sorted**：sorted=True或不传。
- **return_counts**：return_counts=False。
- **dim**：dim=0。
2. 不满足条件1时，将会调用torch.unique，所有的约束需参考torch.unique。参照：[torch.unique](https://docs.pytorch.org/docs/stable/generated/torch.unique.html)，[aclnnUniqueDim](https://gitcode.com/cann/ops-nn/blob/master/index/unique_with_counts_ext2/docs/aclnnUniqueDim.md)
## 性能说明
- 在上述条件基础上，小shape（N<=5500）建议直接使用torch.unique。

## 调用示例
```python
import torch,torch_npu
import numpy as np
from mx_driving_cloud import unique_dim

# Generate random input tensor with shape [2008350, 3] and int32 dtype
input_data = torch.randint(low=0, high=1290, size=(2008350, 3), dtype=torch.int32).npu()

# Call unique_dim with return_inverse=True
unique_tensor, inverse_indices = unique_dim(
    input_data,
    sorted=True,
    return_inverse=True,
    return_counts=False,
    dim=None
)

print(f"Unique tensor shape: {unique_tensor.shape}")
print(f"Inverse indices shape: {inverse_indices.shape}")
```