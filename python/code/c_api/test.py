#!/usr/bin/env python3

import hello

assert hello.i1 == 10
assert hello.i2 == -2000
assert hello.i3 == 10000
assert hello.i4 == 100000
assert hello.say() == 10
p = hello.Person()
print(p)

p = hello.People(last='a', first='b')
print(p.name())
#  print(p.set_first("foobar", 12))
#  print(p.name())
print(hello.say_self())
#  print(help(hello.say_self))
