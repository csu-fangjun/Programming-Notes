
CXXFLAGS := -std=c++17 # for c++17
CXXFLAGS += -pthread

srcs := $(shell find . -type f -name "*.cc" -printf "%f ")
targets :=$(srcs:%.cc=%)

all: $(targets)

%: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

run-%: %
	./$<

.PHONY: clean
clean:
	$(RM) $(targets)

