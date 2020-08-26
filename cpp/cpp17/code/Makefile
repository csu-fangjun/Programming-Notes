
# change it to c++17; c++1z is a synonym for c++17
CXXFLAGS := -std=c++1z

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

