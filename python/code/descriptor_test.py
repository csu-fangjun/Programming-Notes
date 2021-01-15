#!/usr/bin/env python3


class Descr1:

    def __set_name__(self, owner, name):
        self.private_name = f'_{name}'


class Descr2:

    def __set_name__(self, owner, name):
        self.private_name = f'_{name}'

    def __get__(self, obj, objtype=None):
        return 10


class Descr3:

    def __set_name__(self, owner, name):
        self.private_name = f'_{name}'

    def __get__(self, obj, objtype=None):
        # in our case, `obj` is an instance of `Obj1`
        # and `objtype` is `Obj1`
        assert isinstance(obj, objtype)
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)


class Descr4:

    def __set_name__(self, owner, name):
        self.private_name = f'_{name}'

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)


class Obj1:
    d1 = Descr1()
    d2 = Descr2()
    d3 = Descr3()
    d4 = Descr4()


def test_obj1():
    o = Obj1()
    # d1, d2, and d3 are attribute of the class
    print(Obj1.__dict__)
    # {'__module__': '__main__', 'd1': <__main__.Descr1 object at 0xxxx>,
    #  'd2': <__main__.Descr2 object at 0xxxx>,
    #  'd3': <__main__.Descr3 object at 0xxxx>,
    #  'd4': <__main__.Descr3 object at 0xxxx>,
    #  '__dict__': <attribute '__dict__' of 'Obj1' objects>,
    #  '__weakref__': <attribute '__weakref__' of 'Obj1' objects>,
    #  '__doc__': None}

    # d1, d2, and d3 are NOT attribute of the instance !
    print(o.d1)  # <__main__.Descr1 object at 0x7f84366a9760>

    # Descr1 does not have `__get__` method, so o.d1
    # is of type <class '__main__.Descr1'>
    print(type(o.d1))
    print(o.d1.__dict__)  # {'private_name': '_d1'}
    print(Descr1.__dict__)

    # Descr2 has a method `__get__`, so o.d2 returns 10
    print(type(o.d2))  # <class 'int'>

    print(o.d2)  # 10

    print(o.__dict__)  # {}

    o.d3 = 100  # it calls Descr3.__set__(o.d3, o, 100)
    print(o.d3)  # 100
    print(o.__dict__)  # {'_d3': 100}

    o.d = 200
    print(o.d)  # 200
    print(o.__dict__)  # {'_d3': 100, 'd': 200}

    o.d1 = 11
    print(type(o.d1))  # <class 'int'>
    print(o.__dict__)  # {'_d3': 100, 'd': 200, 'd1':11}

    o.d2 = 22
    print(type(o.d2))  # <class 'int'>
    print(o.__dict__)  # {'_d3': 100, 'd': 200, 'd1':11, 'd2': 22}

    # Note since Descr3 has a method `__set__`, it is a data descriptor!
    # o.__dict__ will not contain `d3` !!!
    o.d3 = 33
    print(type(o.d3))  # <class 'int'>
    print(o.__dict__)  # {'_d3': 33, 'd': 200, 'd1':11, 'd2': 22}

    o.__dict__['d3'] = -3
    print(o.d3)  # it is still 33
    print(o.__dict__)  # {'_d3': 33, 'd': 200, 'd1':11, 'd2': 22, 'd3': -3}

    o.d4 = -4

    # NOTE: since Descr4 has no `__get__` method, o.d4 returns an object of Descr4 !!!
    # o.d4 = -4 does not add `d4` to `o.__dict__` since d4 has the method `__set__`!!!
    print(o.d4)  # <__main__.Descr4 object at 0xxxx>
    print(o.__dict__)  # {'_d3': 33, 'd': 200, 'd1': 11, 'd2': 22, 'd3': -3, '_d4': -4}
    print(o._d4)  # -4


if __name__ == '__main__':
    test_obj1()
