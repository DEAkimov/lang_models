import torch


def get_non_pad_mask(seq, pad_idx):
    assert seq.dim() == 2
    return seq.ne(pad_idx).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    """ For masking out the padding part of key sequence. """
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask
