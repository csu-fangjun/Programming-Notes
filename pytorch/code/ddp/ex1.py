#!/usr/bin/env python3

# see https://pytorch.org/tutorials/intermediate/dist_tuto.html

import os
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank: int, world_size: int):
    print(f'rank: {rank}')


def init_process(rank: int, world_size: int, fn):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    fn(rank, world_size)


if __name__ == '__main__':
    world_size = 3
    processes = []
    mp.set_start_method('spawn')
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
