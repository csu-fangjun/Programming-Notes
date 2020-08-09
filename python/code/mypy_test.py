#!/usr/bin/env python3

from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Callable
from typing import TypeVar


# with mypy --disallow-untyped-defs, it also
# prints an error message for "sub".
#
# Without the option, every not annotated functions
# will pass the static type check, since there is no
# type for checking.
def sub(i):
    pass


def iterable_of_str(s: Iterable[str]) -> None:
    pass


def list_of_str(s: List[str]) -> None:
    pass


def tuple_of_one_str(s: Tuple[str]) -> None:
    pass


def tuple_of_many_str(s: Tuple[str, ...]) -> None:
    pass


def add(i: int, a: int) -> int:
    return i + a


def either_int_or_str(s: Union[int, str]) -> None:
    pass


# it is the same as Union[str, None]
def either_str_or_none(s: Optional[str]) -> None:
    pass


def callable_fun(f: Callable[[int, int], int]):
    pass


# this is a type alias
VectorInt = Tuple[int, ...]


# It accepts a tuple of int, e.g., (1,), (1,2)
def vector_of_int(v: VectorInt) -> None:
    pass


# S can be either int, str of List[int]; euqivalent to Union
S = TypeVar('S', int, str, List[int])


# it accepts either int, or str, of list of ints.
def int_or_str_or_list_of_int(x: S) -> None:
    pass


def main() -> None:
    print(add.__annotations__)
    s = add(2, 2)
    list_of_str(['a', 'b'])
    list_of_str(['a', 'b', 'c', 'd', 'e'])
    print(list_of_str.__annotations__)
    #  list_of_str(('a', 'b')) # error, we pass a Tuple[str, str]
    #  tuple_of_one_str(['a', 'b']) # error
    #  tuple_of_one_str(('a', 'b'))  # error, It requires Tuple[str], but we pass Tuple[str, str]
    tuple_of_one_str(('a', ))
    print(tuple_of_one_str.__annotations__)

    tuple_of_many_str(('s', ))
    tuple_of_many_str(('s', 'h'))
    print(tuple_of_many_str.__annotations__)

    iterable_of_str(('a', 'b'))
    iterable_of_str(['a', 'b'])
    #  iterable_of_one_str(['a', 1]) # error
    either_int_or_str('s')
    either_int_or_str(1)
    #  either_int_or_str(None) # error
    either_str_or_none(None)
    either_str_or_none('s')
    #  either_str_or_none(1)  # error
    print(callable_fun.__annotations__)
    callable_fun(add)
    #  callable_fun(either_int_or_str)  # error

    print(vector_of_int.__annotations__)
    vector_of_int((2, ))
    vector_of_int((2, 3))
    #  vector_of_int((2, 3, 's')) # error
    print(int_or_str_or_list_of_int.__annotations__)
    int_or_str_or_list_of_int(1)
    int_or_str_or_list_of_int('2')
    int_or_str_or_list_of_int([1, 2, 3])
    #  int_or_str_or_list_of_int((1, 2, 3)) # error


if __name__ == '__main__':
    main()
