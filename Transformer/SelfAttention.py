import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super(SelfAttention).__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.d_model, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn_softmax = nn.Softmax()
        attn = self.dropout()
        output = torch.matmul(attn_softmax, v)
        return output, attn
