
CXXFLAGS := -std=c++11

c_srcs := $(shell find . -type f -name "*.c" -printf "%f ")
cc_srcs := $(shell find . -type f -name "*.cc" -printf "%f ")

c_targets :=$(c_srcs:%.c=%)
cc_targets :=$(cc_srcs:%.cc=%)

targets := $(c_targets) $(cc_targets)

all: $(targets)

%: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

run-%: %
	./$<

.PHONY: clean
clean:
	$(RM) $(targets)

