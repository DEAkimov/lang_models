import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    def __init__(self, d_model, n_head, d_head, dropout):
        super(Attention, self).__init__()
        # dimension parameters
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head

        # projectors
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.rel_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.out_net = nn.Linear(n_head * d_head, d_model, bias=False)

        # dropout, LN, scale
        self.dropout = nn.Dropout(dropout)
        self.drop_att = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / (d_head ** 0.5)

    @staticmethod
    def _rel_shift(x):
        batch, query, key, n_head = x.size()
        zero_pad = torch.zeros((batch, query, 1, n_head), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=2)
        x_padded.view(batch, key + 1, query, n_head)
        x = x_padded[:, 1:].view_as(x)
        # TODO: zero triu?
        return x

    def forward(self, x, relative_pos, r_w_bias, r_r_bias, attn_mask=None, memory=None):
        # x - [Batch, Time, Dim]
        # r - [Time, Dim]
        batch, q_len, r_len = x.size(0), x.size(1), relative_pos.size(0)
        # cat with memory and project into Q, K, V
        if memory is not None:
            cat = torch.cat([memory, x])
            w_heads = self.qkv_net(cat)
        else:
            w_heads = self.qkv_net(x)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        w_head_q = w_head_q[:, -q_len:]
        r_head_k = self.rel_net(relative_pos)
        k_len = w_head_k.size(1)

        # split heads
        w_head_q = w_head_q.view(batch, q_len, self.n_head, self.d_head)
        w_head_k = w_head_k.view(batch, k_len, self.n_head, self.d_head)
        w_head_v = w_head_v.view(batch, k_len, self.n_head, self.d_head)
        r_head_k = r_head_k.view(r_len, self.n_head, self.d_head)

        # compute attention score
        rw_head_q = w_head_q + r_w_bias
        rr_head_q = w_head_q + r_r_bias
        # [batch x q_time x n_head x d_head] @ [batch x k_time x n_head x d_head] ->
        # -> [batch x q_time x k_time x n_head]
        ac = torch.einsum('bind,bjnd->bijn', [rw_head_q, w_head_k])
        bd = torch.einsum('bind,jnd->bijn', [rr_head_q, r_head_k])
        bd = self._rel_shift(bd)
        attn_score = ac + bd
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None:
            attn_score = attn_score.float().masked_fill(
                attn_mask[:, None, :, None], -float('inf')  # TODO: check this attn mask
            ).type_as(attn_score)
        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.drop_att(attn_prob)

        # compute attention vector (context)
        context = torch.einsum('bijn,bjnd->bind', [attn_prob, w_head_v])
        context = context.view(batch, q_len, self.n_head * self.d_head)

        # compute output
        attn_out = self.out_net(context)
        attn_out = self.dropout(attn_out)
        output = self.layer_norm(x + attn_out)
        return output
