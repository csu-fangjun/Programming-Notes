#!/usr/bin/env python3

# Author: Fangjun Kuang <csukuangfj@gmail.com>

import torch
from typing import Tuple
import math


def my_multihead_self_attention(
        in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
        out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int,
        training: bool, dropout: float) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Args:
      in_proj_weight:
        A 2-D tensor of shape (3*embed_dim, embed_dim). It is a concatenation of
        the weight from three linear layers:

            - One for query
            - One for key
            - One for value

      in_proj_bias:
        A 1-D tensor of shape (3*embed_dim,)
      out_proj_weight:
        A 2-d tensor of shape (embed_dim, embed_dim)
      out_proj_bias:
        A 1-D tensor of shape (embed_dim,)
      q:
        Query. A 3-D tensor of shape (target_len, batch_size, embed_dim)
      k:
        Key. A 3-D tensor of shape (src_len, batch_size, embed_dim)
      v:
        Value. A 3-D tensor of shape (src_len, batch_size, embed_dim)
      num_heads:
        num_heads in the MultiheadAttention.
      training:
        True if it is training. Used for dropout.
      dropout:
        The dropout probability. Valid only if training is True.
    '''
    assert in_proj_weight.ndim == 2
    assert in_proj_bias.ndim == 1
    assert out_proj_weight.ndim == 2
    assert out_proj_bias.ndim == 1
    assert q.ndim == 3
    assert k.ndim == 3
    assert v.ndim == 3

    embed_dim = in_proj_weight.size(1)

    assert in_proj_weight.shape == (3 * embed_dim, embed_dim)
    assert in_proj_bias.shape == (3 * embed_dim,)
    assert out_proj_weight.shape == (embed_dim, embed_dim)
    assert out_proj_bias.shape == (embed_dim,)
    assert q.size(2) == embed_dim
    assert k.size(2) == embed_dim
    assert k.shape == v.shape
    assert q.size(1) == k.size(1)

    target_len = q.size(0)
    batch_size = q.size(1)

    src_len = k.size(0)
    assert k.size(1) == batch_size

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim

    # First, apply the projection
    q_in_proj_weight = in_proj_weight[:embed_dim]
    k_in_proj_weight = in_proj_weight[embed_dim:2 * embed_dim]
    v_in_proj_weight = in_proj_weight[2 * embed_dim:]

    q_in_proj_bias = in_proj_bias[:embed_dim]
    k_in_proj_bias = in_proj_bias[embed_dim:(2 * embed_dim)]
    v_in_proj_bias = in_proj_bias[2 * embed_dim:]

    q = torch.matmul(q, q_in_proj_weight.t()) + q_in_proj_bias
    k = torch.matmul(k, k_in_proj_weight.t()) + k_in_proj_bias
    v = torch.matmul(v, v_in_proj_weight.t()) + v_in_proj_bias

    q = q.contiguous().view(target_len, batch_size * num_heads, head_dim)
    k = k.contiguous().view(src_len, batch_size * num_heads, head_dim)
    v = v.contiguous().view(src_len, batch_size * num_heads, head_dim)

    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    assert q.shape == (batch_size * num_heads, target_len, head_dim)
    assert k.shape == (batch_size * num_heads, src_len, head_dim)
    assert v.shape == (batch_size * num_heads, src_len, head_dim)

    scale = 1 / math.sqrt(head_dim)
    q = q * scale

    # bmm and matmul are equivalent here
    attn_output_weights = torch.matmul(q, k.transpose(1, 2))
    #  attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert attn_output_weights.shape == \
            (batch_size*num_heads, target_len, src_len)
    attn_output_weights = torch.nn.functional.softmax(attn_output_weights,
                                                      dim=-1)

    # NOTE: dropout is applied to attn_output_weights
    attn_output_weights = torch.nn.functional.dropout(attn_output_weights,
                                                      p=dropout,
                                                      training=training)

    attn_output = torch.bmm(attn_output_weights, v)

    assert attn_output.shape == (batch_size * num_heads, target_len, head_dim)
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.contiguous().view(target_len, batch_size,
                                                embed_dim)
    attn_output = torch.matmul(attn_output,
                               out_proj_weight.t()) + out_proj_bias

    attn_output_weights = attn_output_weights.view(batch_size, num_heads,
                                                   target_len, src_len)

    attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
    #  attn_output_weights = attn_output_weights.mean(dim=1)

    return attn_output, attn_output_weights


@torch.no_grad()
def main():
    torch.manual_seed(20210410)
    embed_dim = 8
    num_heads = 2
    dropout = 0.1
    self_attn = torch.nn.MultiheadAttention(embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            dropout=dropout)
    self_attn.train()
    torch.nn.init.uniform_(self_attn.in_proj_bias)
    torch.nn.init.uniform_(self_attn.out_proj.bias)

    state_dict = self_attn.state_dict()

    assert state_dict['in_proj_weight'].shape == (3 * embed_dim, embed_dim)
    assert state_dict['in_proj_bias'].shape == (3 * embed_dim,)
    assert state_dict['out_proj.weight'].shape == (embed_dim, embed_dim)
    assert state_dict['out_proj.bias'].shape == (embed_dim,)

    seq_len = 3
    batch_size = 2
    query = torch.rand(seq_len, batch_size, embed_dim)

    torch.manual_seed(20210410)
    p = self_attn(query=query, key=query, value=query)

    torch.manual_seed(20210410)
    attn_output, attn_output_weights = my_multihead_self_attention(
        in_proj_weight=state_dict['in_proj_weight'],
        in_proj_bias=self_attn.in_proj_bias,
        out_proj_weight=self_attn.out_proj.weight,
        out_proj_bias=self_attn.out_proj.bias,
        q=query,
        k=query,
        v=query,
        num_heads=num_heads,
        training=self_attn.training,
        dropout=dropout)

    assert torch.allclose(p[0], attn_output)
    assert torch.allclose(p[1], attn_output_weights)
    print(p[1])


if __name__ == '__main__':
    main()
