import torch
import torch.nn as nn
from .pos import PositionalEncoding
from .attention import Attention
from .mlp import MLP


class Block(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout):
        super(Block, self).__init__()

        self.attention = Attention(d_model, n_head, d_head, dropout)
        self.mlp = MLP(d_model, 4 * d_model, dropout)

    def forward(self,
                layer_input, relative_pos,
                content_bias, position_bias,
                attn_mask=None, memory=None):
        layer_output = self.attention(
            layer_input, relative_pos,
            content_bias, position_bias,
            attn_mask, memory
        )
        layer_output = self.mlp(layer_output)
        return layer_output
