
# run this file with:
#
#  make -f subst.mk

s := $(subst ee,EE,feet on the street)
$(info $(s)) # fEEt on the strEEt

# note that the leading space in ` EE` is NOT ignored
# be careful of whitespaces in the argument!
s := $(subst ee, EE,feet on the street)
$(info $(s)) # f EEt on the str EEt

# since arguments are separated with comma `,`, to replace
# the space with comma, use substitution
comma := ,
empty :=
# note that there are two $(empty) !!!
space := $(empty) $(empty)
f := feet on the stree
s := $(subst $(space),$(comma),feet on the street)
$(info $(s)) # feet,on,the,street

s := $(subst $(space),$(comma),feet  on the  street) # note there are two spaces!
$(info $(s)) # feet,,on,the,,street  # so we get two commas!

# now replace comma with space
s := $(subst $(comma),$(empty),feet,on,the,street) # empty is not space! empty is nothing!
$(info $(s)) # feetonthestreet
s := $(subst $(comma),$(space),feet,on,the,street)
$(info $(s)) # feet on the street

srcs := a.c a.c.c
objs := $(subst .c,.o,$(srcs)) # do not use this!
$(info $(objs)) # a.o a.o.o

objs := $(srcs:.c=.o) # use this one!
$(info $(objs)) # a.o a.c.o



all:
