import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

        self.encoder = Encoder()

        self.decoder = Decoder()

        self.projection = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_attn = self.encoder(enc_inputs)

        dec_outputs, dec_attn, dec_enc_attn = self.encoder(dec_inputs, enc_inputs, enc_outputs)

        dec_logit = self.projection(dec_outputs)

        return
