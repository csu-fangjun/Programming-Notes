
srcs := $(shell find . -type f -name "*.cu" -printf "%f ")
targets :=$(srcs:%.cu=%)
$(info $(targets))

NVCCFLAGS := -std=c++11 -arch=compute_50

%: %.cu
	nvcc $(NVCCFLAGS) -o $@ $<

clean:
	$(RM) $(targets)
