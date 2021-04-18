#!/usr/bin/env python3


def main():
    # construction
    # set() is a set, while {} is a dict.
    assert set() != {}

    s = {}
    assert isinstance(s, dict)

    s = {1}
    assert isinstance(s, set)

    s = set()
    assert isinstance(s, set)

    a = 'a1a1'
    s = {i for i in a}
    assert s == {'a', '1'}

    s = {1, 2, 'a', 'b'}

    # member testing
    assert 1 in s

    assert 3 not in s  # preferred
    assert not 3 in s  # don't use it!

    assert 'a' in s
    assert 'c' not in s

    # operations
    s = {1, 'a'}
    q = {'a', 2}

    a = s - q
    assert a == {1}
    assert a == set.difference(s, q)

    a = q - s
    assert a == {2}
    assert a == set.difference(q, s)

    a = s | q
    assert a == {1, 2, 'a'}
    assert a == {1, 'a', 2}  # the order does not matter
    assert a == set.union(s, q)
    assert a == set.union(s, q, s, q, s)

    a = s & q
    assert a == {'a'}
    assert a == set.intersection(s, q)
    assert a == set.intersection(s, q, s, q, s)

    a = s ^ q  # in s or in q, but not both
    assert a == (q ^ s)
    assert a == {1, 2}
    assert a == set.symmetric_difference(s, q)
    assert a == set.symmetric_difference(q, s)

    a = (s | q) - (s & q)
    assert a == {1, 2}

    assert not set.isdisjoint(s, q)

    assert not set.issubset(s, q)
    assert not set.issubset(q, s)

    assert not set.issuperset(s, q)
    assert not set.issuperset(q, s)

    # operations
    s = {1, 2}

    # |=
    s.update((3,))
    assert s == {1, 2, 3}

    # |=
    s.update({4, 5})
    assert s == {1, 2, 3, 4, 5}

    # -=
    s.difference_update((3,))
    assert s == {1, 2, 4, 5}

    s.add(3)
    assert s == {1, 2, 3, 4, 5}

    s.remove(3)  # raise an error if 3 is not present
    assert s == {1, 2, 4, 5}

    s.discard(3)  # ok, even if 3 is not present
    s.discard(4)
    s.discard(5)
    assert s == {1, 2}

    s.clear()
    assert s == set()


if __name__ == '__main__':
    main()
