import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    def __init__(self,
                 n_head, d_model, k,
                 dropout):
        super(Attention, self).__init__()
        # dimension parameters
        self.n_head = n_head
        self.d_model = d_model
        self.d_v = (d_model // n_head) * k

        # projectors
        self.query_projector = nn.Linear(d_model, d_model * k, bias=False)
        self.key_projector = nn.Linear(d_model, d_model * k, bias=False)
        self.value_projector = nn.Linear(d_model, d_model * k, bias=False)
        self.relative_projector = nn.Linear(d_model, d_model * k, bias=False)
        self.context_projector = nn.Linear(d_model * k, d_model, bias=False)

        # biases from (C) and (D)
        self.u = nn.Parameter(torch.Tensor(self.d_v))
        self.v = nn.Parameter(torch.Tensor(self.d_v))

        # attention
        self.attn_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

        # layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def _attn(self, query, key, value, relative, mask):
        ac = torch.bmm(query + self.u, key.transpose(1, 2))
        bd = torch.bmm(query + self.v, relative.transpose(1, 2))
        attn_logits = ac + bd
        attn_logits = attn_logits / math.sqrt(key.size(-1))
        # attn_logits: [b*h, q_t, k_t]
        if mask is not None:
            kv_time = mask.size()[-1]
            temp = attn_logits[:, :, -kv_time:].masked_fill(mask, -float('inf'))
            attn_logits[:, :, -kv_time:] = temp

        attn_weights = self.softmax(attn_logits)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.bmm(attn_weights, value)
        return context

    @staticmethod
    def project_view(tensor, layer, sizes):
        temp = layer(tensor).view(*sizes)  # [b, t, heads, features]
        temp = temp.permute(2, 0, 1, 3).contiguous()  # [heads, b, t, f]
        temp = temp.view(sizes[2] * sizes[0], sizes[1], sizes[3])  # [b * h, t, f]
        return temp

    def forward(self, query, key_value, relative, mask=None):
        # {query, key, value} in form [batch, time, features]
        residual = query
        query_batch, query_time, _ = query.size()  # query_batch == kv_batch
        kv_batch, kv_time, _ = key_value.size()
        q_size = (query_batch, query_time, self.n_head, self.d_v)
        kv_size = (kv_batch, kv_time, self.n_head, self.d_v)

        query = self.project_view(query, self.query_projector, q_size)
        key = self.project_view(key_value, self.key_projector, kv_size)
        value = self.project_view(key_value, self.value_projector, kv_size)
        relative = self.project_view(relative, self.relative_projector, kv_size)

        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1)
        context = self._attn(query, key, value, relative, mask)
        context = context.view(self.n_head, kv_batch, query_time, self.d_v)
        context = context.permute(1, 2, 0, 3).contiguous()  # [b, q_time, h, f]
        context = context.view(kv_batch, query_time, -1)  # [b, q_time, h * f]

        output = self.context_projector(context)
        output = self.dropout(output + residual)
        output = self.layer_norm(output)
        return output
