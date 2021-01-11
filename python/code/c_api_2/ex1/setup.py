#!/usr/bin/env python3

# refer to https://docs.python.org/3/extending/building.html#building

from distutils.core import setup, Extension

my_module = Extension(
    'spam',
    sources=['spammodule.cc'],
    language='c++',
    extra_compile_args=['-std=c++11'],
)

setup(
    name='spam_package',  # package name
    version='1.0',
    description='my spam module',
    ext_modules=[my_module])
'''
running build
running build_ext
building 'spam' extension
creating build
creating build/temp.linux-x86_64-3.8
gcc \
  -pthread \
  -Wno-unused-result \
  -Wsign-compare \
  -DNDEBUG \
  -g \
  -fwrapv \
  -O3 \
  -Wall \
  -fPIC \
  -I/path/to/3.8.6/include/python3.8 \
  -c spammodule.cc \
  -o build/temp.linux-x86_64-3.8/spammodule.o
creating build/lib.linux-x86_64-3.8
g++ \
  -pthread \
  -shared \
  -L/path/to/3.8.6/lib \
  -Wl,-rpath=/path/to/3.8.6/lib \
  build/temp.linux-x86_64-3.8/spammodule.o \
  -L/path/to/3.8.6/lib \
  -o build/lib.linux-x86_64-3.8/spam.cpython-38-x86_64-linux-gnu.so
'''
'''
$ python3-config --extension-suffix
.cpython-38-x86_64-linux-gnu.so
'''
