
s := $(findstring a,a b c)
ifeq ($(s), a)
$(info "a b c contains a") # choose this one
else
$(info "a b c does NOT contain a")
endif

# s will be empty
s := $(findstring a,b c)
ifeq ($(s), a)
$(info "b c contains a")
else
$(info "b c does NOT contain a") # choose this one
endif

ifeq ($(s),)
$(info s is empty) # choose this one
else
$(info s is NOT empty)
endif

all:

