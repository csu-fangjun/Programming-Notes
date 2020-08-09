# make -f words.mk

f := foo bar baz
s := $(words $(f))
$(info $(s)) # 3

all:
