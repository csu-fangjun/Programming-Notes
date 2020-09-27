#!/usr/bin/env python3

# refer to
#  https://www.python.org/dev/peps/pep-0484/
#  https://www.python.org/dev/peps/pep-3107/
#  https://www.pythonsheets.com/notes/python-typing.html


def hello(s: str, bb: "any thing", dd, cc: 'world' = 10) -> None:
    pass


def hello2(s):
    pass


def main():
    annotations = hello.__annotations__
    assert annotations.get('s') == str
    assert annotations.get('return') == None
    assert annotations.get('bb') == 'any thing'
    assert annotations.get('cc') == 'world'
    assert 'dd' not in annotations
    print(annotations)


if __name__ == '__main__':
    main()
