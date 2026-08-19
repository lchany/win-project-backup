## conv_bn_eval
### 接口原型
```python
mx_driving_cloud.conv_bn_eval(Tensor running_var, Tensor bias_on_the_fly, Tensor bn_weight,
    Tensor bn_bias, Tensor running_mean, Tensor weight_on_the_fly, float eps)
```
### 功能描述
将mmengine框架中efficient_conv_bn_eval_forward函数中的一部分计算逻辑整合为融合算子。
### 参数说明
- `running_var(Tensor)`:入参变量,shape为[B],数据类型为float
- `bias_on_the_fly(Tensor)`:入参变量,shape为[B],数据类型为float
- `bn_weight(Tensor)`:入参变量,shape为[B],数据类型为float
- `bn_bias(Tensor)`:入参变量,shape为[B],数据类型为float
- `running_mean(Tensor)`:入参变量,shape为[B],数据类型为float
- `weight_on_the_fly(Tensor)`:入参变量,shape为[B,C,H,W],数据类型为float, 约束为C*H*W不超过22000
- `eps(float)`:入参变量,计算逻辑中使用防止除0
### 返回值(inplace)
- `bias_on_the_fly(Tensor)`：计算结果，数据类型为`float`。
- `weight_on_the_fly(Tensor)`：计算结果，数据类型为`float`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving_cloud import conv_bn_eval
N,C,H,W = 64,3,7,7
running_var = torch.rand((N,), dtype=torch.float32).npu()
bn_weight = torch.rand((N,), dtype=torch.float32).npu()
bn_bias = torch.rand((N,), dtype=torch.float32).npu()
running_mean = torch.rand((N,), dtype=torch.float32).npu()
bias_on_the_fly = torch.rand((N,), dtype=torch.float32).npu()
weight_on_the_fly = torch.rand((N,C,H,W), dtype=torch.float32).npu()
eps = 0.000001
conv_bn_eval(running_var, bias_on_the_fly, bn_weight, bn_bias, running_mean, weight_on_the_fly, eps)

```