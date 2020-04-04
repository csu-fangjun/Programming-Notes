
CXXFLAGS := -std=c++11
CXXFLAGS += -I$(HOME)/open-source/benchmark/include
LDFLAGS := -L$(HOME)/open-source/benchmark/build/src -lbenchmark
LDFLAGS += -pthread

srcs := $(shell find . -type f -name "*.cc" -printf %f)
targets :=$(srcs:%.cc=%)

all: $(targets)

%: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
