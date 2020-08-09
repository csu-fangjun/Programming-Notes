# make -f reverse.mk

# f := one two three
f := one two

# the following `reverse` adds a leading space
# reverse = $(if $(1),$(call reverse,$(wordlist 2,$(words $(1)),$(1)))) $(firstword $(1))

# the following `reverse` is recommended.
reverse = $(if $(wordlist 2,2,$(1)),$(call reverse,$(wordlist 2,$(words $(1)),$(1))) $(firstword $(1)),$(1))

s := $(call reverse,$(f))
$(info $(s))

all:
