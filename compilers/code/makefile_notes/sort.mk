# make -f sort.mk

f := foo bar baz bar foo

s := $(sort $(f))
$(info $(s)) # bar baz foo

all:
