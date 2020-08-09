# make -f word.mk

f := foo bar baz
s := $(word 1,$(f))
$(info $(s)) # foo

s := $(word 2,$(f))
$(info $(s)) # bar

s := $(word 3,$(f))
$(info $(s)) # baz

s := $(word 4,$(f))
$(info $(s)) # an empty line is printed; s is empty;

all:
