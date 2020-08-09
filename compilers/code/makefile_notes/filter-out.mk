# make -f filter-out.mk

foo := a.cc b.o a.c d.cc libm.so

s := $(filter-out %.cc,$(foo))
$(info $(s)) # b.o a.c libm.so

s := $(filter-out %.cc %.c,$(foo))
$(info $(s)) # b.o libm.so

all:
