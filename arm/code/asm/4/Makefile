# Copyright 2020. All Rights Reserved.
# Author: fangjun.kuang@gmail.com (Fangjun Kuang)

PREFIX := aarch64-linux-gnu-
CXX := $(PREFIX)g++

CXXFLAGS := -std=c++11

srcs := main.cc
srcs += inc_static_variable.s

.PHONY: all clean run

all: main

main: main.cc $(srcs)
	$(CXX) $(CXXFLAGS) -o $@ $^

run: main
	qemu-aarch64 -L $(HOME)/software/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc  ./main

clean:
	$(RM) main *.o
