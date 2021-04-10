#!/usr/bin/env python3

# Author: Fangjun Kuang <csukuangfj@gmail.com>

import torch


@torch.no_grad()
def main():
    embed_dim = 6
    num_heads = 2
    self_attn = torch.nn.MultiheadAttention(embed_dim=embed_dim,
                                            num_heads=num_heads)

    seq_len = 2
    batch_size = 3
    query = torch.rand(seq_len, batch_size, embed_dim)
    attn_mask = torch.rand(1, batch_size * num_heads, seq_len, seq_len)
    print(attn_mask.shape)
    p = self_attn(query=query, key=query, value=query, attn_mask=attn_mask)
    print(p[0].shape, p[1].shape)


if __name__ == '__main__':
    main()
