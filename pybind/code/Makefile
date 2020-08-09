
srcs := hello.cc
srcs += handle_test.cc
srcs += object_test.cc
srcs += attr_test.cc
srcs += array_test.cc
srcs += vector_test.cc
srcs += basics_test.cc

objs := $(srcs:%.cc=%.o)

suffix := $(shell python3-config --extension-suffix)

CXXFLAGS := -std=c++11
CXXFLAGS += $(shell python3 -m pybind11 --includes)

name := hello$(suffix)

all: $(name)

%.o: %.cc header.h
	$(CXX) -c -fPIC $(CXXFLAGS) -o $@ $<

$(name): $(objs)
	$(CXX) -shared -o $@ $^

.PHONE: clean test

test: $(name)
	python3 -c "import hello; print(hello.add(1, 2))"

clean:
	-$(RM) $(name) *.o
