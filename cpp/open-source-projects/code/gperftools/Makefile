
LDFLAGS := -L/home/fangjunkuang/software/gperftools/lib -ltcmalloc
LDFLAGS += -Wl,-rpath,/home/fangjunkuang/software/gperftools/lib
CXXFLAGS := -I/home/fangjunkuang/software/gperftools/include
CXXFLAGS += -std=c++11

all:

%: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

run: main
	HEAPPROFILE=a.prof LD_PRELOAD=/home/fangjunkuang/software/gperftools/lib/libtcmalloc.so ./main

draw:
	pprof -unit MB -http 0.0.0.0:1234 ./a.prof.0001.heap

.PHONY: run draw clean
clean:
	$(RM) a.prof.* main tcmalloc_test
