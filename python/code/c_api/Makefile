

CXXFLAGS := -std=c++11 # -Wall -Wextra
CXXFLAGS += $(shell python3-config --includes)
LDFLAGS += $(shell python3-config --ldflags)

suffix := $(shell python3-config --extension-suffix)
suffix := .so

# srcs := $(shell find . -name "*.cc" | xargs -I{} basename {})
srcs := $(shell find . -name "*.cc" -printf "%P\n")
objs := $(srcs:%.cc=%.o)


.PHONY: all
all: hello$(suffix)

%.o: %.cc
	$(CXX) -c -fPIC $(CXXFLAGS) -o $@ $<

hello$(suffix): $(objs)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)


.PHONY: clean
clean:
	$(RM) hello.*so $(objs)
