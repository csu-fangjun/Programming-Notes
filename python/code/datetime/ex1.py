#!/usr/bin/env python3

import datetime


def main():
    a = datetime.date.today()
    assert isinstance(a, datetime.date)
    assert isinstance(a.year, int)
    assert isinstance(a.month, int)
    assert isinstance(a.day, int)

    print(dir(a))
    assert isinstance(a.isoformat(), str)  # 2021-06-06
    print(a.toordinal())
    print(datetime.date.fromordinal(10))  # 0001-01-10
    print(datetime.date.fromordinal(32))  # 0001-02-01
    print(a.ctime())  # Sat Jun 26 00:00:00 2021

    # See https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
    print(a.strftime('%y-%m-%d'))  # 21-06-26
    print(a.strftime('%Y-%m-%d'))  # 2021-06-26

    a = datetime.datetime.utcnow()
    print(a)  # 2021-06-26 07:30:11.353549
    assert isinstance(a, datetime.datetime)
    print(dir(a))
    print(a)


if __name__ == '__main__':
    main()
