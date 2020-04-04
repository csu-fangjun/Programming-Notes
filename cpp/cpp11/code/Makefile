
CXXFLAGS := -std=c++14

srcs := $(shell find . -type f -name "*.cc" -printf %f)
targets :=$(srcs:%.cc=%)

all: $(targets)

%: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<


.PHONY: clean
clean:
	$(RM) $(targets)

