# Copyright 2020. All Rights Reserved.
# Author: fangjun.kuang@gmail.com (Fangjun Kuang)

PREFIX := aarch64-linux-gnu-
CXX := $(PREFIX)g++

CXXFLAGS := -std=c++11

asm_srcs := $(shell find . -name "*.S" -printf "%f ")
c_srcs := $(shell find . -name "*.c" -printf "%f ")
cc_srcs := $(shell find . -name "*.cc" -printf "%f ")

asm_targets = $(asm_srcs:%.S=%)
c_targets = $(c_srcs:%.c=%)
cc_targets = $(cc_srcs:%.cc=%)

targets := $(asm_targets)
targets += $(c_targets)
targets += $(cc_targets)

.PHONY: all clean run

all: $(targets)

%: %.S
	$(CXX) $(CXXFLAGS) -o $@ $^

%: %.c
	$(CXX) $(CXXFLAGS) -o $@ $^

%: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $^

run-%: %
	qemu-aarch64 -L $(HOME)/software/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc  $<

clean:
	$(RM) $(targets)

