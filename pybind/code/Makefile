srcs := hello.cc

suffix := $(shell python3-config --extension-suffix)

CXXFLAGS := -std=c++11
CXXFLAGS += -shared -fPIC
CXXFLAGS += $(shell python3 -m pybind11 --includes)

name := hello$(suffix)

all: $(name)

$(name): $(srcs)
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONE: clean test

test: $(name)
	python3 -c "import hello; print(hello.add(1, 2))"

clean:
	-$(RM) $(name)
