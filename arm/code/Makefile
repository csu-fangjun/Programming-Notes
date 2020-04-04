
PREFIX := aarch64-linux-gnu-
CXX := $(PREFIX)g++

CXXFLAGS := -std=c++11

.PHONY: all clean run

srcs := inline_asm.cc

all: main

main: main.cc $(srcs)
	$(CXX) $(CXXFLAGS) -o $@ $^

run: main
	qemu-aarch64 -L $(HOME)/software/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc  ./main

clean:
	$(RM) main *.o *.s
