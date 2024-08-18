import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim, dk, dv, dropout=0.0):
        super().__init__()

        self.scale = dk ** -0.5
        self.q = nn.Linear(dim, dk)
        self.k = nn.Linear(dim, dk)
        self.v = nn.Linear(dim, dv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [1, 4, 2]——>[batch_size, num_token, seq_length]
        q = self.q(x)  # q:[1, 4, 3]——>[batch_size, num_token, dk]
        print('q:', q)
        k = self.k(x)  # k:[1, 4, 3]——>[batch_size, num_token, dk]
        print('k:', k)
        v = self.v(x)  # k:[1, 4, 4]——>[batch_size, num_token, dv]
        print('v:', v)

        # attn_score = softmax((q*k^T)*(dk^-0.5))*v
        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        print('attn:', attn)
        # attn是一个torch.Tensor对象，所以可以直接调用softmax
        attn = attn.softmax(dim=-1)
        print('attn_score:', attn)
        attn = self.dropout(attn)
        print('attn_dropout:', attn)
        output = torch.matmul(attn, v)

        return output


# 模型输入参数包括：输入数据的维度, 映射之后的dk,dv维度大小
model = SelfAttention(dim=2, dk=3, dv=4)
# 随机生成一个输入样本，1是batch_size, 4是指4个token，2是每个token的长度
input_data = torch.randn(1, 4, 2)
print('input_data:', input_data)
final_output = model(input_data)
print('final_output:', final_output)

