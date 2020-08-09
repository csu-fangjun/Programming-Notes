# make -f wordlist.mk

f := foo bar baz
s := $(wordlist 1,2,$(f))
$(info $(s)) # foo bar

s := $(wordlist 1,3,$(f))
$(info $(s)) # foo bar baz

s := $(wordlist 1,4,$(f))
$(info $(s)) # foo bar baz

s := $(wordlist 2,3,$(f))
$(info $(s)) # bar baz

s := $(wordlist 3,3,$(f))
$(info $(s)) # baz

s := $(wordlist 2,1,$(f))
$(info $(s)) # s is empty

all:
