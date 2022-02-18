#!/usr/bin/env python3

import torch

x = torch.tensor(
    [
        [
            [1, 2, 3],
            [0, 3, 9],
        ],
        [
            [-3, 5, -2],
            [10, 2, 5],
        ],
    ]
)

print(x.shape)  # (2, 2, 3)

y = torch.tensor(
    [
        [
            [0, -2, -3],
            [0, -3, -9],
            [5, -3, 3],
        ],
        [
            [3, -2, 2],
            [0, 6, -8],
            [1, 3, 5],
        ],
    ]
)
print(y.shape)  # (2, 3, 3)

#  (2, 2, 1, 3)
#  (2, 1, 3, 3)
s = x.unsqueeze(2) + y.unsqueeze(1)  # (2, 1, 3, 3)
print(s.shape)  # (2, 2, 3, 3)
print(s)
print(x.unsqueeze(2) + (y * 0).unsqueeze(1))
