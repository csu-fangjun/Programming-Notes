#!/bin/bash

lex ex.l
gcc lex.yy.c -lfl

echo "login name is username" | ./a.out

# it should print
#  login name is foo
# where foo is the output of `id -n -u`
rm a.out lex.yy.c

