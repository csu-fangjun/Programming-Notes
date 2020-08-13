#!/usr/bin/env python3

import re


def test1():
    p = re.compile('ab*')
    assert isinstance(p, re.Pattern)

    # match always begins from index 0
    m = p.match('abc')  # matches, so do not return None
    assert isinstance(m, re.Match)
    assert m
    assert m.group() == 'ab'

    m = p.match('abbbc')
    assert m.group() == 'abbb'

    m = p.match('deabc')  # match starts from 0, so there is no match here
    assert m is None

    m = p.match('def')
    assert m is None

    # start is 2 because 'a'  is at position 2
    # end is 5 because the last character that does not match the regular expression is 5
    m = p.search('deabbc')
    assert m.start() == 2
    assert m.end() == 5
    assert m.span() == (2, 5)
    assert m.group() == 'abb'


def main():
    test1()


if __name__ == '__main__':
    main()