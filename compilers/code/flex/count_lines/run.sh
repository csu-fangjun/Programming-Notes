#!/bin/bash

set -e

# count number of lines and number of characters

# lex count.l
lex count3.l
gcc lex.yy.c -lfl


printf "12\n345 \n"
printf "12\n345 \n" | ./a.out

# It should print
#
#  num of lines: 2
#  num of characters: 5

rm -f lex.yy.c a.out
