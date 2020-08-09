#!/bin/bash

# we only need
#
# (1) calc.l, for the scanner
# (2) calc.y, for the parser

# it generates calc.tab.c and calc.tab.h
bison -d calc.y

flex calc.l

gcc -o calc calc.tab.c lex.yy.c -ll

