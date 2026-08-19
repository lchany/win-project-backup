## qr
### 接口原型
```python
mx_driving_cloud.linalg.qr(Tensor A) -> Tensor Q, Tensor R
```
### 功能描述
对矩阵A进行QR分解，输出分解后的矩阵Q和矩阵R
### 参数说明
- `A(Tensor)`：输入待分解张量A，数据类型为`float32`。shape 为`[M, N]`。
### 返回值
- `Q(Tensor)`：分解所得张量Q，数据类型为`float32`。
- `R(Tensor)`：分解所得张量R，数据类型为`float32`。
### 约束说明
- M <= 9984
- N <= 9984
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch
import mx_driving_cloud
A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32).npu()
Q, R = mx_driving_cloud.linalg.qr(A)
```