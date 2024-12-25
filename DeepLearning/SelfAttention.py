import numpy as np
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.scale = dim ** -0.5
        self.projections = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])  # q, k, v
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q, k, v = [proj(x) for proj in self.projections]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output


# 模型输入参数包括：输入数据的维度
model = SelfAttention(dim=2)
# 随机生成一个输入样本，1是batch_size, 4是指4个token，2是每个token的长度
input_data = torch.randn(1, 4, 2)
print('input_data:', input_data)
final_output = model(input_data)
print('final_output:', final_output)
