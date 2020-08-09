# make -f filter.mk

foo := a.cc b.o a.c d.cc libm.so

s := $(filter %.cc,$(foo))
$(info $(s)) # a.cc d.cc

s := $(filter %.cc %.c,$(foo))
$(info $(s)) # a.cc a.c d.cc

all:
