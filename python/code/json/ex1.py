#!/usr/bin/env python3

import json
import os

# See
# https://docs.python.org/3/library/json.html


def main():
    a = json.dumps(['foo'])
    assert isinstance(a, str)
    assert a == '["foo"]'

    a = json.dumps(['foo', {'a': 1, 'b': 'hello'}])
    assert isinstance(a, str)
    assert a == '["foo", {"a": 1, "b": "hello"}]'

    obj = {'a': 10, 'b': 'hello'}
    with open('a.txt', 'w') as f:
        json.dump(obj, f, ensure_ascii=False)

    with open('a.txt') as f:
        obj = json.load(f)
        assert isinstance(obj, dict)
        assert obj == {'a': 10, 'b': 'hello'}
    os.remove('a.txt')


if __name__ == '__main__':
    main()
