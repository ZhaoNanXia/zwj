import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, dim, dk, dv, dropout=0.0):
        super().__init__()
        self.num_head = num_head
        self.dim = dim
        self.dk = dk
        self.dv = dv

        self.q = nn.Linear(dim, dk)
        self.k = nn.Linear(dim, dk)
        self.v = nn.Linear(dim, dv)
        self.scale = 1 / math.sqrt(dim // num_head)
        self.fc = nn.Linear(num_head * dv, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_token, dim_in = x.shape
        dk = self.dim // self.num_head
        dv = self.dim // self.num_head
        q = self.q(x).reshape(batch_size, num_token, self.num_head, dk).transpose(1, 2)
        k = self.k(x).reshape(batch_size, num_token, self.num_head, dk).transpose(1, 2)
        v = self.v(x).reshape(batch_size, num_token, self.num_head, dv).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(2, 3)) * self.scale

        attn = attn.softmax(dim=-1)

        attn_score = torch.matmul(attn, v)

        attn_score = attn_score.transpose(1, 2).reshape(batch_size, num_token, self.dim)

        output = self.fc(attn_score)
        return output


input_data = torch.randn(1, 4, 2)
model = MultiHeadAttention(num_head=2, dim=2, dk=2, dv=3)
final_output = model(input_data)
print(final_output)



