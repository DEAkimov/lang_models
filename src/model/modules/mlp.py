from torch.nn.functional import relu
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 d_in, d_hid,
                 dropout):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_in),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_in)

    def forward(self, x):
        mlp = self.mlp(x)
        output = self.layer_norm(mlp + x)
        return output
