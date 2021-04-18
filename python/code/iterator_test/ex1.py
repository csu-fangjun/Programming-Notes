#!/usr/bin/env python


# Counter is an iterable since it has a `__iter__` method;
# it is also an iterator because it has a `__next__` method.
#
# It is not good to place `__iter__` and `__next__` in the same
# class since `__next__` changes the class status and it's not re-entrant.
class Counter(object):

    def __init__(self, n):
        self.i = -1
        self.n = n

    def __iter__(self):
        # the returned object has to implement `next(self)`
        return self

    def __next__(self):
        self.i += 1
        if self.i >= self.n:
            raise StopIteration
        return self.i


def test1():
    c = Counter(3)
    d = list(c)
    assert d == [0, 1, 2]
    e = list(c)
    assert e == []  # c has already been consumed!

    c = Counter(2)
    a = iter(c)
    assert next(a) == 0
    assert a.__next__() == 1
    #  print(next(a)) # it raises StopIteration


class MyIteratable:

    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return MyIterator(self)


class MyIterator:

    def __init__(self, iterable):
        self.iterable = iterable
        self.i = -1

    def __next__(self):
        self.i += 1
        if self.i >= self.iterable.n:
            raise StopIteration
        print('return ', self.i)
        return self.i

    def __contains__(self, i):
        # __contains__ is for the op `in`
        # if we do not define it, it will iterator through the iterator
        # to find the value
        if self.i < i and i < self.iterable.n:
            return True

    def __iter__(self):
        return self


def test2():
    c = MyIteratable(3)
    #  d = list(c)
    #  assert d == [0, 1, 2]
    #  e = list(c)
    #  assert e == [0, 1, 2] # c is still there !
    m = MyIterator(c)
    assert 1 in m
    assert 1 in m


class MyCallable:

    def __init__(self):
        self.i = -1

    def __call__(self):
        self.i += 1
        if self.i > 5:
            raise StopIteration
        return self.i


def test3():
    # iter can also accepts a callable and a sentinel
    #
    # When the return value equal to the `sentinel`, it
    # raises StopIteration, so the for loop ends
    for i in iter(MyCallable(), 2):
        pass
    d = list(iter(MyCallable(), 2))
    assert d == [0, 1]

    # It's return value will never be -1, so
    # it ends only when it raises a StopIteration
    d = list(iter(MyCallable(), -1))
    assert d == [0, 1, 2, 3, 4, 5]


class MyItem:
    # this is the old way
    # when the for loop encounters IndexError, it exits the loop
    def __getitem__(self, i):
        if i > 3:
            raise IndexError
        return i


def test4():
    for i in MyItem():
        pass
    d = list(MyItem())
    assert d == [0, 1, 2, 3]


if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()
