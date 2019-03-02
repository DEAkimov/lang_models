import torch
import torch.nn as nn
from .pos import PositionalEncoding
from .attention import Attention
from .mlp import MLP


class Block(nn.Module):
    def __init__(self,
                 n_head, d_model, k,
                 dropout):
        super(Block, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.attention = Attention(n_head, d_model, k, dropout)
        self.mlp = MLP(d_model, 4 * d_model, dropout)

    def forward(self,
                layer_input, content_bias, position_bias,
                non_pad_mask, slf_attn_mask,
                memory):
        query = layer_input
        if memory is not None:
            key_value = torch.cat([memory, layer_input], dim=1)
        else:
            key_value = layer_input
        pos = torch.arange(
            key_value.size(1) - 1, -1, -1,
            dtype=torch.float32,
            device=key_value.device
        )
        pos = self.positional_encoding(pos, layer_input.size(0))
        output = self.attention(
            query, key_value, pos,
            content_bias, position_bias,
            slf_attn_mask
        )
        output *= non_pad_mask
        output = self.mlp(output)
        output *= non_pad_mask
        return output


class SuperBlock(nn.Module):
    def __init__(self,
                 n_head, d_model, k,
                 n_steps, n_topics,
                 dropout):
        super(SuperBlock, self).__init__()
        self.n_steps = n_steps
        self.step_embedding = nn.Embedding(n_steps, d_model)
        self.conditioning = nn.Linear(n_topics, d_model)
        self.block = Block(n_head, d_model, k, dropout)

    def forward(self,
                layer_input, condition, memory,
                content_bias, position_bias,
                non_pad_mask, slf_attn_mask):
        def tensor(t):
            return torch.tensor(
                [[t] * time] * batch,
                dtype=torch.long,
                device=device
            )
        batch, time = layer_input.size()[:2]
        device = layer_input.device
        if condition is not None:
            condition = self.conditioning(condition)
            output = layer_input + condition
        else:
            output = layer_input
        if memory is None:
            memory = [None for _ in range(self.n_steps)]
        new_memory = []
        for step, mem in zip(range(self.n_steps), memory):
            step_emb = self.step_embedding(tensor(step))
            output = output + step_emb
            new_memory.append(output)
            output = self.block(
                output, content_bias, position_bias,
                non_pad_mask, slf_attn_mask, mem
            )
        return output, new_memory
