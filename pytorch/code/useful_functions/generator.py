#!/usr/bin/env python3

import torch


def test1():
    a = torch.Generator()
    b = torch.Generator()

    assert a.initial_seed() == b.initial_seed()
    a.manual_seed(10)
    a.initial_seed() == 10

    # return a random number that can be used as a seed
    # Note it also changes the seed of the generator!
    s = a.seed()
    assert a.initial_seed() != 10
    assert a.initial_seed() == s

    a.manual_seed(10)
    state = a.get_state()

    a.manual_seed(100)
    assert a.initial_seed() == 100
    a.set_state(state)

    # Restore its state, so its seed is still 10
    assert a.initial_seed() == 10


def test2():
    # PyTorch has a global default_generator
    # see torch/random.py
    torch.manual_seed(10)
    assert torch.default_generator.initial_seed() == 10
    assert torch.default_generator.device.type == 'cpu'

    state = torch.get_rng_state()
    torch.manual_seed(100)

    assert torch.initial_seed() == 100
    s = torch.seed()  # it calls torch.default_generator.seed()
    assert torch.initial_seed() == s

    torch.set_rng_state(state)
    assert torch.initial_seed() == 10


def main():
    test1()
    test2()


if __name__ == '__main__':
    main()
