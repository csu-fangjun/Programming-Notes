

.PHONY: all


CXXFLAGS := $(shell pkg-config --cflags protobuf)
CXXFLAGS += -std=c++11 -g
LDFLAGS := $(shell pkg-config --libs protobuf)

PROTOBUF_LIB_DIR := $(shell pkg-config --variable=libdir protobuf)
LDFLAGS += "-Wl,-rpath,$(PROTOBUF_LIB_DIR)"
LDFLAGS += -pthread

# all: hello

# use this one be I want to see the content of .pb.cc
all: abc/addressbook.pb.cc hello

%.o: %.cc
	$(CXX) -c $(CXXFLAGS) $< -o $@

hello: hello.o abc/addressbook.pb.o
	$(CXX) -o $@ $^ $(LDFLAGS)

abc/%.pb.cc: %.proto
	protoc $< --cpp_out=./abc

.PHONY: clean
clean:
	$(RM) hello *.o abc/*.pb.*
