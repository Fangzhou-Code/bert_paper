import numpy as np
import torch

# 创建一个 Parameter 对象
x = torch.nn.parameter.Parameter(torch.randn(3, 4))
print("x:",x)

# 获取 Parameter 对象对应的张量
x_data = x.data

# 将张量转换为 NumPy 数组
x_array = x_data.numpy()

# 遍历数组并读取每个元素
for i in range(x_array.shape[0]):
    for j in range(x_array.shape[1]):
        print("Element at position ({}, {}): {}".format(i, j, x_array[i,j]))