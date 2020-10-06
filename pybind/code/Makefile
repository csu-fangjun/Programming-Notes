
srcs := hello.cc
srcs += handle_test.cc
srcs += object_test.cc
srcs += attr_test.cc
srcs += array_test.cc
srcs += vector_test.cc
srcs += basics_test.cc
srcs += classes.cc
srcs += doc_test.cc

objs := $(srcs:%.cc=%.o)

suffix := $(shell python3-config --extension-suffix)

CXXFLAGS := -std=c++11
CXXFLAGS += $(shell python3 -m pybind11 --includes)

name := hello$(suffix)

OS := $(shell uname -s)
ifeq ($(OS), Darwin)
SHARED_LIB_FLAG := -dynamiclib -Wl,-flat_namespace
SHARED_LIB_FLAG += -Wl,-undefined,suppress
# either suppress or dynamic_lookup is fine
# SHARED_LIB_FLAG += -Wl,-undefined,dynamic_lookup
else
SHARED_LIB_FLAG := -shared
endif

all: $(name)

%.o: %.cc header.h
	$(CXX) -c -fPIC $(CXXFLAGS) -o $@ $<

$(name): $(objs)
	$(CXX) $(SHARED_LIB_FLAG) -o $@ $^

.PHONE: clean test

test: $(name)
	python3 -c "import hello; print(hello.add(1, 2))"

clean:
	-$(RM) $(name) *.o
