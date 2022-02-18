#!/usr/bin/env python3

import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import torch
import torch.nn as nn

import datetime


def get_data(idx: int):
    ans = torch.tensor([idx, idx + 1, idx + 2], dtype=torch.float32, requires_grad=True)
    return ans


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(3, 2)
        self.linear1 = nn.Linear(3, 2)
        self.linear2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor, idx: int):
        if idx == 0:
            y = self.linear0(x)
        elif idx == 1:
            y = self.linear1(x)
        elif idx == 2:
            y = self.linear2(x)
        else:
            raise ValueError("idx should be 0, 1 or 2")
        return y.sum()


def run(rank: int, world_size: int):
    print(f"world_size: {world_size}")
    device = torch.device("cuda", rank)

    model = Model()
    model.to(device)
    print(f"model: {model}")
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(100):
        print(f"iter: {i}")
        data = get_data(rank + i).to(device)
        print(f"rank: {rank}, data: {data}")

        optimizer.zero_grad()
        y = model(data, rank)
        print("y", y)
        y.backward()
        optimizer.step()

    print(f"rank {rank} done")


def init_process(rank: int, world_size: int, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(0, 5)
    )
    fn(rank, world_size)


if __name__ == "__main__":
    print(f"dist.is_available: {dist.is_available()}")
    world_size = 3
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
