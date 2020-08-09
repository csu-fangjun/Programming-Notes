# make -f lastword.mk

f := foo bar baz
s := $(lastword $(f))
$(info $(s)) # baz

s := $(word $(words $(f)),$(f))
$(info $(s)) # baz

g :=
s := $(lastword $(g))
$(info $(s)) # s is empty

all:
