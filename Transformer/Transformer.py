import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size, output_dim):
        super(Embeddings, self).__init__()

        self.embed = nn.Embedding(vocab_size, output_dim)

    def forward(self, x):
        return self.embed(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding()


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

        self.encoder = Encoder()  # 编码层

        self.decoder = Decoder()  # 解码层

        self.projection = nn.Sequential(
            nn.Linear(d_model, vocab_size, bias=False),
            nn.Softmax(dim=1)
        )  # 输出层
        # d_model是解码层输出的维度；最后还需要进行一个softmax将结果映射为每个词的概率大小

    def forward(self, enc_inputs, dec_inputs):
        # 输入参数包括两部分：编码层输入和解码层输入——[batch_size, ]
        enc_outputs, enc_self_attn = self.encoder(enc_inputs)

        dec_outputs, dec_self_attn, dec_enc_attn = self.encoder(dec_inputs, enc_inputs, enc_outputs)

        output = self.projection(dec_outputs)

        return output


if __name__ == '__main__':
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
