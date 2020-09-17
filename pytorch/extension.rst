
Extensions
==========


hello world (cpu)
-----------------

.. code-block::
  :caption: hello.cc
  :language: cpp

  #include <torch/extension.h>

  torch::Tensor sigmoid(torch::Tensor z) {
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sigmoid", &sigmoid, "sigmoid test");
  }

.. code-block::
  :caption: setup.py
  :language: python

  from setuptools import setup, Extension
  from torch.utils import cpp_extension

  setup(name='hello',
        ext_modules=[cpp_extension.CppExtension('hello', ['hello.cc'])],
        cmdclass={'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)})

The output of ``python setup.py build``::

    running build
    running build_ext
    building 'hello' extension
    creating build
    creating build/temp.linux-x86_64-3.7
    gcc -pthread -Wno-unused-result -Wsign-compare \
      -DNDEBUG -g -fwrapv -O3 -Wall -fPIC \
      -I/xxx/py37/lib/python3.7/site-packages/torch/include \
      -I/xxx/py37/lib/python3.7/site-packages/torch/include/torch/csrc/api/include \
      -I/xxx/py37/lib/python3.7/site-packages/torch/include/TH \
      -I/xxx/py37/lib/python3.7/site-packages/torch/include/THC \
      -I/xxx/include/python3.7m \
      -c hello.cc \
      -o build/temp.linux-x86_64-3.7/hello.o \
      -DTORCH_API_INCLUDE_EXTENSION_H \
      -DTORCH_EXTENSION_NAME=hello \
      -D_GLIBCXX_USE_CXX11_ABI=0 \
      -std=c++14

    g++ \
      -pthread \
      -shared \
      -L/xxx/lib \
      build/temp.linux-x86_64-3.7/hello.o \
      -L/xxx/py37/lib/python3.7/site-packages/torch/lib \
      -lc10 \
      -ltorch \
      -ltorch_cpu \
      -ltorch_python \
      -o build/lib.linux-x86_64-3.7/hello.cpython-37m-x86_64-linux-gnu.so


hello world (cuda)
------------------

Change ``setup.py``. Replace ``cpp_extension.CppExtension`` with ``cpp_extension.CUDAExtension``.

The output of ``python setup.py build``::

    -I/usr/local/cuda/include


    -L/usr/local/cuda/lib64 \
    -lcudart \
    -lc10_cuda \
    -ltorch_cuda

Internals
---------

``torch.utils`` is in ``torch/utils/__init__.py``.

To find where ``torch`` is installed::

  import os.path
  import torch
  print(os.path.dirname(torch.__file__))

``cpp_extension`` is in ``torch.utils/cpp_extension.py``.

If ``ninja`` is availabe, it is used by default. The environment
variable ``MAX_JOBS`` can be used to limit the number of CPUs
for ``ninja``. ``ninja --version`` can be used to check
the availabilitity of ``ninja``.

.. code-block::

  print(torch._C._GLIBCXX_USE_CXX11_ABI) # True or False


References
----------

- `<https://pytorch.org/tutorials/advanced/cpp_extension.html>`_
