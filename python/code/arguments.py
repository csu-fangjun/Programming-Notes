#!/usr/bin/env python3


def keyword_only(*, a, b):
    return a + b


def main():
    # TypeError: keyword_only() takes 0 positional arguments but 2 were given
    #  c = keyword_only(3, 4)
    c = keyword_only(a=3, b=4)
    assert c == 7


if __name__ == "__main__":
    main()
