#!/usr/bin/env python3

# refer to https://docs.python.org/3/extending/building.html#building

from distutils.core import setup, Extension

my_module = Extension(
    'integer',
    sources=['integer.cc'],
    language='c++',
    extra_compile_args=['-std=c++11', '-O0'],
)

setup(
    name='integer_package',  # package name
    version='1.0',
    description='my integer module',
    ext_modules=[my_module])
