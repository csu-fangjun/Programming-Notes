CXXFLAGS = -I$(HOME)/open-source/cereal-1.3.0/include
CXXFLAGS += -std=c++11

hello: hello.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	$(RM) hello
