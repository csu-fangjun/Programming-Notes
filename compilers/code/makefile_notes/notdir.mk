# make -f notdir.mk

f = src/a.cc b.cc foo/bar/d.cc abc def/gh/
s := $(notdir $(f))
$(info $(s)) # a.cc b.cc d.cc abc
# note that def/gh/  is removed

all:
