
NAME := hello
SUFFIX := $(shell python3-config --extension-suffix)
LIB_NAME := $(NAME)$(SUFFIX)

CUDA_HOME := $(shell which nvcc | xargs dirname | xargs dirname)

#USE_CXX11_ABI := $(shell python3 -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
USE_CXX11_ABI := $(shell python3 -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
TORCH_INSTALL_DIR := $(shell python3 -c 'import os; import torch; print(os.path.dirname(torch.__file__))')

CXXFLAGS := $(shell python3-config --cflags)

CXXFLAGS += -I$(TORCH_INSTALL_DIR)/include
CXXFLAGS += -I$(TORCH_INSTALL_DIR)/include/torch/csrc/api/include
CXXFLAGS += -I$(TORCH_INSTALL_DIR)/include/TH
CXXFLAGS += -I$(TORCH_INSTALL_DIR)/include/THC
CXXFLAGS += -std=c++14
CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=$(USE_CXX11_ABI)

# if we use #include <torch/torch.h>, then we need the following macro
# if we use #include <torch/extension.h>, then it is not required
CXXFLAGS += -DTORCH_API_INCLUDE_EXTENSION_H
CXXFLAGS += -DTORCH_EXTENSION_NAME=hello

CXXFLAGS += -fPIC
CXXFLAGS += -Wno-unknown-pragmas # disable omp warnings

LDFLAGS := -shared
LDFLAGS += $(shell python3-config --ldflags)
LDFLAGS += -L$(TORCH_INSTALL_DIR)/lib
LDFLAGS += -lc10 -ltorch -ltorch_cpu -ltorch_python
LDFLAGS += -Wl,-rpath,$(TORCH_INSTALL_DIR)/lib

# HAS_CUDA := $(shell test -d $(CUDA_HOME) && echo yes || echo no)
HAS_CUDA := $(shell python3 -c 'import torch; print("yes" if torch.cuda.is_available() else "no")')
$(info "has cuda: $(HAS_CUDA)")

ifeq ($(HAS_CUDA),yes)
CXXFLAGS += -I$(CUDA_HOME)/include
LDFLAGS += -L$(CUDA_HOME)/lib64
LDFLAGS += -lcudart -lc10_cuda -ltorch_cuda
LDFLAGS += -Wl,-rpath,$(CUDA_HOME)/lib64
endif

.PHONY: all clean
all: $(LIB_NAME)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(LIB_NAME): hello.o
	$(CXX) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) -rf build $(LIB_NAME) *.o
	$(RM) -rf dist foobar my_hello.egg-info
