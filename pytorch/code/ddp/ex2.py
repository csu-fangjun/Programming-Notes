#!/usr/bin/env python3

# see https://pytorch.org/tutorials/intermediate/dist_tuto.html

import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch

import datetime


def run(rank: int, world_size: int):
    print(f'rank: {rank}, {dist.get_rank()}')
    print(f'world_size: {world_size}, {dist.get_world_size()}')
    device = torch.device('cuda', rank)
    a = torch.tensor([0, 1, 2], dtype=torch.int32, device=device) + rank

    # all gather example
    # We have to pre-allocate space for entries in `res`.
    dist.barrier()
    res = [torch.empty_like(a) for _ in range(world_size)]
    # res[i] is from node with rank == i
    dist.all_gather(res, a)

    for i in range(world_size):
        assert torch.all(torch.eq(res[i].cpu(), torch.tensor([0, 1, 2]) + i))

    # all reduce example. Note that `a` is modified in place
    dist.barrier()
    # a contains the sum from all nodes
    dist.all_reduce(a, op=dist.ReduceOp.SUM)
    expected = torch.zeros_like(a).cpu()
    for i in range(world_size):
        expected += torch.tensor([0, 1, 2]) + i
    assert torch.all(torch.eq(a.cpu(), expected))

    # reduce example
    a = torch.tensor([1, 2], device=device) + rank
    dist.barrier()
    # the node with rank==0 gets the result.
    # Other node's a is not changed.
    dist.reduce(a, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        expected = torch.zeros_like(a).cpu()
        for i in range(world_size):
            expected += torch.tensor([1, 2]) + i
        assert torch.all(torch.eq(a.cpu(), expected))
    else:
        assert torch.all(torch.eq(a.cpu(), torch.tensor([1, 2]) + rank))

    # broadcast example
    a = torch.tensor([1, 2], device=device) + rank
    dist.barrier()
    # send the value `a` on node with rank 0 to other nodes
    dist.broadcast(a, src=0)
    assert torch.all(torch.eq(a.cpu(), torch.tensor([1, 2])))


def init_process(rank: int, world_size: int, fn):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group('nccl',
                            rank=rank,
                            world_size=world_size,
                            timeout=datetime.timedelta(0, 10))
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
