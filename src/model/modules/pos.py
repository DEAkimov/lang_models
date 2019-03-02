import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()

        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, batch):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = pos_emb.expand(batch, -1, -1)
        return pos_emb


if __name__ == '__main__':
    pe = PositionalEncoding(36)
    pos = torch.arange(2, -1, -1.0)
    pose = pe(pos)
