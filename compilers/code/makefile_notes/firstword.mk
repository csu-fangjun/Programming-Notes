# make -f firstword.mk

f := foo bar baz
s := $(firstword $(f))
$(info $(s)) # foo

s := $(word 1,$(f))
$(info $(s)) # foo

g :=
s := $(firstword $(g))
$(info $(s)) # s is empty

all:
