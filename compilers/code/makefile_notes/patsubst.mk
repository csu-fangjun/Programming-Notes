
# patsubst means pattern subst
# syntax is: $(patsubst pattern,replacement,text)
# % is a wildcard

srcs := a.c.c     cc.c
objs := $(patsubst %.c,%.o,$(srcs)) # note that it eats the spaces!
$(info $(objs)) # a.c.o cc.o

cc_srcs := $(patsubst %.o,%.cc,$(objs))
$(info $(cc_srcs)) # a.c.cc cc.cc

o := $(cc_srcs:%.cc=%.o)
$(info $(o)) # a.c.o cc.o

# the following methods are equivalent
#
# $(var:suffix=replacement)
#
# $(patsubst %suffix,%replacement,$(var))

all:
