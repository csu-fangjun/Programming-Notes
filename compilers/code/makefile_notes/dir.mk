# make -f dir.mk

f = src/a.cc b.cc foo/bar/d.cc abc def/gh/
s := $(dir $(f))
$(info $(s)) # src/ ./ foobar/ ./ def/gh/

all:
