
define print2
$(info arg1: $(1));
$(info arg2: $(2))
endef

# call will set $(1), $(2), ..., for the function print2
$(call print2,hello,world)
# arg1: hello
# arg2: world

# the `=` is optional.
define print22 =
$(info arg1: $(1))
$(info arg2: $(2))
endef
# arg1: hello
# arg2: world

$(eval $(call print22,hello,world))

all:
