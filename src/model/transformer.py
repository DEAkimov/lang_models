import torch
import torch.nn as nn

from model.modules.block import SuperBlock
from model.modules.pos import PositionalEncoding
from model.modules.masks import get_non_pad_mask, get_attn_key_pad_mask, get_subsequent_mask


class WeiredTransformer(nn.Module):
    def __init__(self,
                 dict_size, n_topics,
                 bos_idx, pad_idx,
                 n_blocks, n_steps, n_head, d_model, k,
                 dropout):
        super(WeiredTransformer, self).__init__()
        # vocab sizes
        self.base_vocab = dict_size
        # tokens
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx
        # global content and global positional biases
        d_v = (d_model // n_head) * k
        self.content_bias = nn.Parameter(torch.Tensor(n_head, 1, 1, d_v))
        torch.nn.init.xavier_normal_(self.content_bias)
        self.position_bias = nn.Parameter(torch.Tensor(n_head, 1, 1, d_v))
        torch.nn.init.xavier_normal_(self.position_bias)
        # layers
        self.vocab_embedding = nn.Embedding(
            dict_size + 1,
            d_model,
            padding_idx=pad_idx
        )
        self.pos_embedding = PositionalEncoding(d_model)

        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList([
            SuperBlock(n_head, d_model, k, n_steps, n_topics, dropout)
            for _ in range(n_blocks)
        ])
        self.conditioning = nn.Linear(n_topics, d_model)
        self.last_layer = nn.Linear(d_model, dict_size + 1, bias=False)

        self.last_layer.weight = self.vocab_embedding.weight
        self.logit_scale = (d_model ** -0.5)

    @staticmethod
    def detach_mem(mem):
        return [[step.detach() for step in block] for block in mem]

    def forward(self, input_tensor, condition=None, memory=None):
        # needs to feed encoder input to form encoder-decoder attention mask

        vocab_emb = self.vocab_embedding(input_tensor)
        hidden = vocab_emb
        non_pad_mask = get_non_pad_mask(input_tensor, self.pad_idx)
        self_attn_mask_subseq = get_subsequent_mask(input_tensor)
        self_attn_mask_keypad = get_attn_key_pad_mask(input_tensor, input_tensor, self.pad_idx)
        self_attn_mask = (self_attn_mask_subseq + self_attn_mask_keypad).gt(0)

        if memory is None:
            memory = [None for _ in range(self.n_blocks)]

        # print(non_pad_mask.size(), self_attn_mask.size())

        new_memory = []
        for block, mem in zip(self.blocks, memory):
            hidden, new_mem = block(
                hidden, condition, mem,
                self.content_bias, self.position_bias,
                # non_pad_mask, None
                non_pad_mask, self_attn_mask
            )
            new_memory.append(new_mem)

        if condition is not None:
            hidden = hidden + self.conditioning(condition)
        logits = self.last_layer(hidden) * self.logit_scale
        return logits, self.detach_mem(new_memory)

    # TODO: incremental forward


if __name__ == '__main__':
    # just test if model works
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    t = WeiredTransformer(
        20_000, 12,
        0, 10_000,
        2, 2, 2, 32, 2,
        0.1
    )
    print(count_parameters(t))
    opt = torch.optim.SGD(t.parameters(), 1e-2)
    criterion = torch.nn.NLLLoss()
    log_sm = torch.nn.LogSoftmax(dim=-1)
    inp = torch.tensor([
        range(100)
    ] * 3, dtype=torch.long)

    for _ in range(1000):
        c = None
        m = None
        for j in range(10):
            inp_t = inp[:, j * 5:(j + 1) * 5]
            tgt_t = inp[:, j * 5 + 1:(j + 1) * 5 + 1]

            r, m = t(inp_t, c, m)
            lsm = log_sm(r)

            loss = criterion(
                lsm.contiguous().view(3 * 5, -1),
                tgt_t.contiguous().view(-1)
            )
            print(loss.item())
            loss.backward()
            opt.step()
