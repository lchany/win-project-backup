## scatter_max
### 接口原型
```python
mx_driving_cloud.scatter_max_v3_cloud(Tensor src, Tensor index, Tensor out=None) -> Tensor
```
### 功能描述
在第0维上，将输入张量`src`中的元素按照`index`中的索引进行分散，然后在第0维上取最大值，返回最大值。对于1维张量，公式如下：
$$out_i = max(out_i, max_j(src_j))$$
这里，$i = index_j$。
### 参数说明
- `src(Tensor)`：更新源张量[N,C]，数据类型为`float32`。
- `index(Tensor)`：索引张量[N]，数据类型为`int32`。
- `out(Tensor)`：可选参数，更新后张量，数据类型为`float32`，默认为`None`。
### 返回值
- `out(Tensor)`：更新后的张量，数据类型为`float32`。
### 算子约束
- `index`的维度必须为`1`，`index`第0维的长度必须与`src`第0维的长度相同。
- `index`的取值必须为非负的有效索引值。
- `out`的维度必须与`src`的维度相同，且除第0维外其余维的长度必须与`src`相同。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving_cloud import scatter_max_v3_cloud
src = torch.tensor([[2, 0, 1, 3, 1, 0, 0, 4], [0, 2, 1, 3, 0, 3, 4, 2], [1, 2, 3, 4, 4, 3, 2, 1]], dtype=torch.float32).npu()
index = torch.tensor([0, 2, 0], dtype=torch.int32).npu()
out = src.new_zeros((3, 8))
out = scatter_max_v3_cloud(src, index, out)
```