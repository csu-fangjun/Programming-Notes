#!/usr/bin/env python3

# test self-attention

# In self-attention, key==value==query

import torch

import torch.nn as nn


def test_case1():
    # no key padding mask
    embed_dim = 512
    num_heads = 8
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim
    dropout_p = 0.2

    multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout_p)
    # By default, bias is True, which means the in_proj uses Bias.
    # By default, dropout is 0, nothing is dropped
    #
    # If dropout is not 0, the attention weight is dropped.
    # That is, before multiplying with `value`, the weight is dropped.

    # In the case of self-attention, there is only one tensor
    # for the input projection

    assert multihead_attn.in_proj_weight.shape == (3 * embed_dim, embed_dim)
    assert multihead_attn.in_proj_bias.shape == (3 * embed_dim,)

    batch_size = 5
    seq_len = 20
    key = torch.rand(seq_len, batch_size, embed_dim)

    torch.manual_seed(20210621)  # set seed for dropout
    attn_output, attn_output_weights = multihead_attn(key=key,
                                                      value=key,
                                                      query=key)
    assert attn_output.shape == (seq_len, batch_size, embed_dim)
    tgt_len = seq_len
    src_len = seq_len
    assert attn_output_weights.shape == (batch_size, tgt_len, src_len)

    # Now for my own implementation
    # Step 1, apply input linear projection (with bias=True)
    # key is (seq_len, batch_size, ebmed_dim)
    # in_proj_weight is (3*embed_dim, ebmed_dim)
    q, k, v = (torch.matmul(key, multihead_attn.in_proj_weight.t()) +
               multihead_attn.in_proj_bias).chunk(3, dim=-1)

    # q,k,v is (seq_len, batch_size, embed_dim)

    scaling = float(head_dim)**-0.5
    q = q * scaling

    q = q.contiguous().view(tgt_len, batch_size * num_heads,
                            head_dim).transpose(0, 1)
    # q is (batch_size * num_heads, tgt_len, head_dim)

    k = k.contiguous().view(src_len, batch_size * num_heads,
                            head_dim).transpose(0, 1)
    # k is (batch_size * num_heads, src_len, head_dim)

    v = v.contiguous().view(src_len, batch_size * num_heads,
                            head_dim).transpose(0, 1)
    # v is (batch_size * num_heads, src_len, head_dim)

    attn_output_weights2 = torch.matmul(q, k.transpose(1, 2))
    # attn_output_weights2 is (batch_size * num_heads, tgt_len, src_len)

    # apply softmax
    attn_output_weights2 = torch.nn.functional.softmax(attn_output_weights2,
                                                       dim=-1)

    torch.manual_seed(20210621)  # set seed for dropout

    # apply dropout
    attn_output_weights2 = torch.nn.functional.dropout(attn_output_weights2,
                                                       dropout_p)

    # v is (batch_size * num_heads, src_len, head_dim)
    attn_output2 = torch.matmul(attn_output_weights2, v)

    assert attn_output2.shape == (batch_size * num_heads, tgt_len, head_dim)
    attn_output2 = attn_output2.transpose(0, 1).contiguous().view(
        tgt_len, batch_size, embed_dim)

    # apply output linear proj
    attn_output2 = torch.matmul(
        attn_output2,
        multihead_attn.out_proj.weight.t()) + multihead_attn.out_proj.bias

    # Note, the return attention weight is the average of all heads
    attn_output_weights2 = attn_output_weights2.view(
        batch_size, num_heads, tgt_len, src_len).sum(dim=1) / num_heads

    assert torch.allclose(attn_output, attn_output2)
    assert torch.allclose(attn_output_weights, attn_output_weights2)


def generate_key_padding_mask(seq_lens: torch.Tensor):
    '''
    Args:
      seq_len_list:
        A 1-D int tensor of each sequence before padding
    '''
    max_len = seq_lens.max()
    linear = torch.arange(max_len).unsqueeze(0).repeat(seq_lens.size(0), 1)
    # linear is
    # [
    #  [0, 1, 2, ...]
    #  [0, 1, 2, ...]
    #  [0, 1, 2, ...]
    # ]
    return seq_lens.unsqueeze(-1) <= linear
    # masked positions are True
    # If input is [1, 3, 2], then it returns
    # [
    #  [False, True, True],
    #  [False, False, False],
    #  [False, False, True],


def test_case2():
    # with key padding mask
    embed_dim = 512
    num_heads = 8
    head_dim = embed_dim // num_heads
    assert num_heads * head_dim == embed_dim

    dropout_p = 0.8

    multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout_p)
    batch_size = 3
    seq_len = 5
    key = torch.rand(seq_len, batch_size, embed_dim)
    lens = torch.randint(1, seq_len + 1, size=(batch_size,))
    lens[0] = seq_len  # so that seq_len is always present
    key_padding_mask = generate_key_padding_mask(lens)

    torch.manual_seed(20210621)  # set seed for dropout
    attn_output, attn_output_weights = multihead_attn(
        key=key, value=key, query=key, key_padding_mask=key_padding_mask)

    assert attn_output.shape == (seq_len, batch_size, embed_dim)

    tgt_len = seq_len
    src_len = seq_len

    assert attn_output_weights.shape == (batch_size, tgt_len, src_len)

    q, k, v = (torch.matmul(key, multihead_attn.in_proj_weight.t()) +
               multihead_attn.in_proj_bias).chunk(3, dim=-1)

    scaling = float(head_dim)**-0.5
    q = q * scaling

    q = q.contiguous().view(tgt_len, batch_size * num_heads,
                            head_dim).transpose(0, 1)
    # q is (batch_size * num_heads, tgt_len, head_dim)

    k = k.contiguous().view(src_len, batch_size * num_heads,
                            head_dim).transpose(0, 1)
    # k is (batch_size * num_heads, src_len, head_dim)

    v = v.contiguous().view(src_len, batch_size * num_heads,
                            head_dim).transpose(0, 1)
    # v is (batch_size * num_heads, src_len, head_dim)

    attn_output_weights2 = torch.matmul(q, k.transpose(1, 2))
    # attn_output_weights2 is (batch_size * num_heads, tgt_len, src_len)
    # key_padding_mask is (batch_size, src_len)
    key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
    # key_padding_mask is (batch_size, 1, 1, src_len)

    attn_output_weights2 = attn_output_weights2.view(batch_size, num_heads,
                                                     tgt_len, src_len)

    # Fill it with -inf since `-inf` is changed to 0 after applying softmax
    attn_output_weights2.masked_fill_(key_padding_mask, float('-inf'))

    attn_output_weights2 = attn_output_weights2.view(batch_size * num_heads,
                                                     tgt_len, src_len)

    attn_output_weights2 = torch.nn.functional.softmax(attn_output_weights2,
                                                       dim=-1)

    torch.manual_seed(20210621)  # set seed for dropout

    # apply dropout
    attn_output_weights2 = torch.nn.functional.dropout(attn_output_weights2,
                                                       dropout_p)

    # v is (batch_size * num_heads, src_len, head_dim)
    attn_output2 = torch.matmul(attn_output_weights2, v)

    assert attn_output2.shape == (batch_size * num_heads, tgt_len, head_dim)
    attn_output2 = attn_output2.transpose(0, 1).contiguous().view(
        tgt_len, batch_size, embed_dim)

    # apply output linear proj
    attn_output2 = torch.matmul(
        attn_output2,
        multihead_attn.out_proj.weight.t()) + multihead_attn.out_proj.bias

    # Note, the return attention weight is the average of all heads
    attn_output_weights2 = attn_output_weights2.view(
        batch_size, num_heads, tgt_len, src_len).sum(dim=1) / num_heads

    assert torch.allclose(attn_output, attn_output2)
    assert torch.allclose(attn_output_weights, attn_output_weights2)


def generate_square_subsequent_mask(max_tgt_len, max_src_len):
    # troch.triu extracts a subpart of the input matrix
    a = torch.triu(torch.ones(max_tgt_len, max_src_len) == 1)
    # if max_tgt_len is 5, max_src_len is 3
    # a is
    # [
    #  [True, True, True, True, True]
    #  [False, True, True, True, True]
    #  [False, False, True, True, True]
    # ]
    a = a.transpose(0, 1)
    print(a)
    return
    # Now a is
    # [
    #  [ True, False, False, False, False]
    #  [ True, True, False, False, False]
    #  [ True, True, True, False, False]
    #  [ True, True, True, True, False]
    #  [ True, True, True, True, True]
    # ]
    a = a.float()
    a.masked_fill_(a == 0, float('-inf'))
    a.masked_fill_(a == 1, float(0.0))
    # Now a is
    # [
    #  [ 0, -inf, -inf, -inf, -inf]
    #  [ 0, 0, -inf, -inf, -inf]
    #  [ 0, 0, 0, -inf, -inf]
    #  [ 0, 0, 0, 0, -inf]
    #  [ 0, 0, 0, 0, 0]
    # ]
    # masked positions are set to -inf
    # unmasked positions are set to 0
    # This is an additive mask!
    a = a.repeat(batch_size, 1, 1)
    # Now a is of shape (batch_size, tgt_len, tgt_len)
    return a


def test4():
    # cross attention
    # with
    # key padding mask
    # attn_mask
    #
    # this is the most complicated one
    embed_dim = 8
    num_heads = 2
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim

    kdim = 3
    vdim = 2
    dropout_p = 0.8

    multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout_p,
                                           kdim=kdim,
                                           vdim=vdim)
    batch_size = 3
    src_len = 4
    tgt_len = 5
    key = torch.rand(src_len, batch_size, kdim)
    value = torch.rand(src_len, batch_size, vdim)
    query = torch.rand(tgt_len, batch_size, embed_dim)

    lens = torch.randint(1, src_len + 1, size=(batch_size,))
    lens[0] = src_len  # so that src_len is always present
    key_padding_mask = generate_key_padding_mask(lens)
    attn_mask = generate_square_subsequent_mask(batch_size, tgt_len)

    torch.manual_seed(20210621)  # set seed for dropout
    attn_output, attn_output_weights = multihead_attn(
        key=key,
        value=key,
        query=key,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask)

    assert attn_output.shape == (tgt_len, batch_size, embed_dim)
    assert attn_output_weights.shape == (batch_size, tgt_len, src_len)

    # key is (src_len, batch_size, kdim)
    # value is (src_len, batch_size, vdim)
    # queyr is (tgt_len, batch_size, embed_dim)
    q = torch.matmul(key, multihead_attn.q_proj_weight.t()
                    ) + multihead_attn.in_proj_bias[:embed_dim]
    k = torch.matmul(key, multihead_attn.k_proj_weight.t()
                    ) + multihead_attn.in_proj_bias[embed_dim:2 * embed_dim]
    v = torch.matmul(key, multihead_attn.v_proj_weight.t()
                    ) + multihead_attn.in_proj_bias[2 * embed_dim:]

    scaling = float(head_dim)**-0.5
    q = q * scaling

    q = q.contiguous().view(tgt_len, batch_size * num_heads,
                            head_dim).transpose(0, 1)
    # q is (batch_size * num_heads, tgt_len, head_dim)

    k = k.contiguous().view(src_len, batch_size * num_heads,
                            head_dim).transpose(0, 1)
    # k is (batch_size * num_heads, src_len, head_dim)

    v = v.contiguous().view(src_len, batch_size * num_heads,
                            head_dim).transpose(0, 1)
    # v is (batch_size * num_heads, src_len, head_dim)

    attn_output_weights2 = torch.matmul(q, k.transpose(1, 2))
    # attn_output_weights2 is (batch_size * num_heads, tgt_len, src_len)
    # key_padding_mask is (batch_size, src_len)
    key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
    # key_padding_mask is (batch_size, 1, 1, src_len)

    attn_output_weights2 = attn_output_weights2.view(batch_size, num_heads,
                                                     tgt_len, src_len)


def main():
    generate_square_subsequent_mask(3, 5)
    torch.manual_seed(20210627)
    #  test_case1()
    #  test_case2()
    #  test_case3()


if __name__ == '__main__':
    main()
