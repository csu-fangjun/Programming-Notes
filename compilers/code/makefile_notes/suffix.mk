# make -f suffix.mk

f = src/a.cc b.cc foo/bar/d.cc abc def/gh/
s := $(suffix $(f))
$(info $(s)) # .cc .cc .cc
# note that abc and def/gh/ are removed

all:
