#!/usr/bin/env python3

import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import torch

import datetime


def run(rank: int, world_size: int):
    print(f'world_size: {world_size}')
    device = torch.device('cuda', rank)
    if rank != 0:
        data = [torch.tensor([1], device=device, dtype=torch.float32) for _ in range(world_size)]
    else:
        data = [torch.tensor([1], device=device, dtype=torch.float32) for _ in range(world_size*100)]
    # NOTE: `data` on rank 0 has more entries
    dist.barrier()

    model = torch.nn.Linear(1, 1).to(device)
    model = DDP(model, device_ids=[rank])
    for i, d in enumerate(data):
        model.zero_grad()
        y = model(d)
        y.backward()

    print(f'rank {rank} done')
    # node with rank==0 will exit after timeout (5 seconds)
    # The default timeout is 5 minutes. But it comes into effect
    # only if one of the following environment variable is
    # set:
    #  - NCCL_ASYNC_ERROR_HANDLING
    #  - NCCL_ASYNC_ERROR_HANDLING
    # See https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group


def init_process(rank: int, world_size: int, fn):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group('nccl',
                            rank=rank,
                            world_size=world_size,
                            timeout=datetime.timedelta(0, 5))
    fn(rank, world_size)


if __name__ == '__main__':
    print(f'dist.is_available: {dist.is_available()}')
    world_size = 3
    processes = []
    mp.set_start_method('spawn')
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
