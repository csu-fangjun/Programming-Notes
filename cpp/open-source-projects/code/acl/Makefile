
PREFIX := aarch64-linux-gnu-
CXX := $(PREFIX)g++

ACL_HOME=$(HOME)/open-source/ComputeLibrary

CXXFLAGS := -std=c++11

CXXFLAGS += -I$(ACL_HOME)
CXXFLAGS += -I$(ACL_HOME)/include
CXXFLAGS += -Wl,-rpath=$(ACL_HOME)/build/build

LDFLAGS := -L$(ACL_HOME)/build/build -larm_compute
LDFLAGS += -larm_compute_core

all: hello

%: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) hello
