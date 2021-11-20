#!/usr/bin/env python3

# https://docs.python.org/3/reference/datamodel.html?highlight=iadd#emulating-numeric-types


class MyInteger:
    def __init__(self):
        self.v = 0

    def __iadd__(self, v):
        """
        +=
        """
        self.v = self.v + v
        return self

    def __isub__(self, v):
        """
        -=
        """
        self.v = self.v - v
        return self

    def __imul__(self, v):
        """
        *=
        """
        self.v = self.v * v
        return self

    def __itruediv__(self, v):
        """
        *=
        """
        self.v = self.v / v
        return self

    def __repr__(self):
        s = "{}".format(self.v)
        return s


def test_MyInteger():
    i = MyInteger()
    assert i.v == 0
    i += 3
    assert i.v == 3
    i += 2
    assert i.v == 5

    i -= 2
    assert i.v == 3

    i *= 5
    assert i.v == 15

    i /= 10
    assert i.v == 1.5
    print(i)


class MyList:
    def __init__(self):
        self.v = []

    def __getitem__(self, i):
        return self.v[i]

    def __setitem__(self, i, val):
        self.v[i] = val

    def extend(self, val):
        """
        This is not a magic method
        """
        self.v.extend(val)


def main():
    test_MyInteger()

    l = MyList()
    l.extend([0, 1, 2])
    assert l.v == [0, 1, 2]


if __name__ == "__main__":
    main()
