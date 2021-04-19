#!/usr/bin/env python3
import torch


def main():
    # uniformly distributed between [low, high)
    # Note that high is not included
    a = torch.randint(low=0, high=2, size=(30,), dtype=torch.int32)
    assert a.max() == 1
    assert a.min() == 0
    assert a.ndim == 1

    a = torch.randint(0, 2, (2, 3, 4))
    assert a.ndim == 3

    # using a generator
    a = torch.Generator()

    a.manual_seed(100)
    state = a.get_state()
    b = torch.randint(0, 2, (2, 3, 4), generator=a)
    c = torch.randint(0, 2, (2, 3, 4), generator=a)
    assert not torch.all(torch.eq(b, c))

    # after restoring the state, we can reproduce it
    a.set_state(state)
    c = torch.randint(0, 2, (2, 3, 4), generator=a)
    assert torch.all(torch.eq(b, c))

    # we can also reset the seed
    a.manual_seed(100)
    c = torch.randint(0, 2, (2, 3, 4), generator=a)
    assert torch.all(torch.eq(b, c))


if __name__ == '__main__':
    main()
