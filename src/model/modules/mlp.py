from torch.nn.functional import relu
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 d_in, d_hid,
                 dropout):
        super(MLP, self).__init__()
        self.layer_1 = nn.Conv1d(d_in, d_hid, 1)
        self.layer_2 = nn.Conv1d(d_hid, d_in, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_in)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.layer_2(relu(self.layer_1(output)))
        output = output.transpose(1, 2)

        output = self.dropout(output + residual)
        output = self.layer_norm(output)
        return output
