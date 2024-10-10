import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_dim, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = emb_dim // num_heads
        self.scale = head_dim ** -0.5
        # 计算qkv的转移矩阵
        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        # 最终的线性层
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_token, dim_in = x.shape  # [b, n, c]
        print(f'batch_size: {batch_size}, num_token: {num_token}, dim_in: {dim_in}')

        qkv = self.qkv(x).reshape(batch_size, num_token, 3, self.num_heads,
                                  dim_in // self.num_heads).permute(2, 0, 3, 1, 4)
        # [b, n, c] ——> [b, n, 3 * c] ——> [b, n, 3, num_heads, head_dim] ——> [3, b, num_heads, n, head_dim]
        print(f'qkv.shape: {qkv.shape}')
        q, k, v = qkv[0], qkv[1], qkv[2]
        # [b, num_heads, n, head_dim]
        attn_score = torch.matmul(q, k.transpose(3, 2)) * self.scale
        # [b, num_heads, n, head_dim] * [b, num_heads, head_dim, n] ——> [b, num_heads, n, n]
        attn_score = attn_score.softmax(dim=-1)
        print('attn_score.shape:', attn_score.shape)

        attn_weight = torch.matmul(attn_score, v).transpose(1, 2).reshape(batch_size, num_token, dim_in)
        # [b, num_heads, n, n] *  [b, num_heads, n, head_dim] ——> [b, num_heads, n, head_dim]
        # ——> [b, n, num_heads, head_dim] ——> [b, n, dim_in]
        print('attn_weight.shape:', attn_weight.shape)

        output = self.fc(attn_weight)

        return output


# 输入：batch_size为1,token数为4,每个token的维度为2
input_data = torch.randn(1, 4, 2)
print('input_data:', input_data)
model = MultiHeadAttention(num_heads=2, emb_dim=2)
final_output = model(input_data)
print(final_output)
