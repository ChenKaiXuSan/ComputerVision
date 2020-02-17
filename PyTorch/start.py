from __future__ import print_function
import torch

# 创建一个矩阵，未初始化
y = torch.empty(5, 3)
print(y)

# 初始化一个矩阵
x = torch.rand(5, 3)
print(x)

import torch
print(torch.cuda.is_available())

print("hello")

# 创建一个0填充的矩阵，数据类型为long:
z = torch.zeros(5, 3, dtype=torch.long)
print(z)

# 创建tensor并使用现有数据初始化:
w = torch.tensor([5.5, 3])
print(w)

x = x.new_ones(5, 3, dtype=torch.double)      # new_* 方法来创建对象
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 覆盖 dtype!
print(x)                                      #  对象的size 是相同的，只是值和类型发生了变化

print(x.size())

y = torch.rand(5, 3)
print(x + y)

print(x[:, 1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  #  size -1 从其他维度推断
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())