#!/usr/bin/env python3


def test_tuple(*a):
    assert isinstance(a, tuple)
    print(a[0])


def test_dict(**a):
    assert isinstance(a, dict)
    print(a)


def test_tuple_dict(*a, **b):
    assert isinstance(a, tuple)
    assert isinstance(b, dict)
    test_tuple(*a)
    test_dict(**b)


def test_keyword_only(*, foo=1):
    pass


def main():
    test_tuple([1, 2])  # print: [1, 2]
    test_tuple(1)  # print: 1

    test_dict(b=1)  # print: {'b': 1}
    test_dict(b=1, c=2)  # print: {'b': 1, 'c': 2}

    test_tuple_dict(1, 2, a=1)
    # print:
    # 1
    # {'a': 1}

    test_tuple_dict((10, 20), b=100)
    # print:
    # (10, 20)
    # {'b': 100}
    #  test_keyword_only(1)  # error
    #  test_keyword_only(bar=1) # error
    test_keyword_only(foo=1)


if __name__ == '__main__':
    main()
