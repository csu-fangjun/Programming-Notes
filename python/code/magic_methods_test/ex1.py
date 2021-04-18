#!/usr/bin/env python3

# https://docs.python.org/3/reference/datamodel.html?highlight=iadd#emulating-numeric-types


class MyInteger:
    def __init__(self):
        self.v = 0

    def __iadd__(self, v):
        '''
        +=
        '''
        self.v = self.v + v
        return self

    def __repr__(self):
        s = '{}'.format(self.v)
        return s


class MyList:
    def __init__(self):
        self.v = []

    def __getitem__(self, i):
        return self.v[i]

    def __setitem__(self, i, val):
        self.v[i] = val

    def extend(self, val):
        '''
        This is not a magic method
        '''
        self.v.extend(val)


def main():
    i = MyInteger()
    i += 2
    assert i.v == 2

    l = MyList()
    l.extend([0, 1, 2])
    assert l.v == [0, 1, 2]


if __name__ == '__main__':
    main()
