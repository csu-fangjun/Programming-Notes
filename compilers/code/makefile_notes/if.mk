
# make -f if.mk

a :=
ifdef a
$(info "a is defined")
else
$(info "a is not defined")  #  choose this one
endif

foo :=
bar := $(foo)

ifdef bar
$(info "bar is defined")
else
$(info "bar is not defined")  #  choose this one since bar is an imediate variable and is empty
endif

f =
b = $(f)
ifdef b
$(info "b is defined") # choose this one!!!
else
$(info "b is not defined")
endif

use_space = true
ifdef use_space
  $(info "we can put leading **spaces** but NOT leading tabs when it is NOT in rules")
endif

all:
ifdef use_space
	$(info "we can put leading **tabs** but NOT leading spaces when it is in rules")
endif

# now for ifeq
# there are four forms of `ifeq`
#
# ifeq (arg1, arg2)
# ifeq "arg1" "arg2"
# ifeq 'arg1' 'arg2'
# ifeq "arg1" 'arg2'
# ifeq 'arg1' "arg2"

a := hello
ifeq ($(a), hello)
$(info a is hello) # choose this one
else
$(info a is NOT hello)
endif

b := hello

ifeq ($(a),   $(b)) # we can have spaces after the comma!
$(info a == b) # choose this one
else
$(info a != b)
endif

ifeq "$(a)" "$(b)"
$(info a == b) # choose this one
else
$(info a != b)
endif

c := world
ifneq ($(a), $(c))
$(info a != c) # prints: a != c
endif
