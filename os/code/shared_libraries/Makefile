
all: ld_preload_test

ld_preload_test: ld_preload_test.cc my_malloc.cc
	$(CXX) -std=c++11 -shared -fPIC -o libmy_malloc.so my_malloc.cc -ldl
	$(CXX) -std=c++11 -o ld_preload_test ld_preload_test.cc
	LD_PRELOAD=./libmy_malloc.so ./ld_preload_test

.PHONY: clean
clean:
	$(RM) *.so *.o ld_preload_test
