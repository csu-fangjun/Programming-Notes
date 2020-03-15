from setuptools import setup
from setuptools import Extension

import pybind11

# refer to https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension
ext_modules = [
    Extension(name='hello',
              sources=['hello.cc'],
              extra_compile_args=['-std=c++11'],
              include_dirs=[pybind11.get_include()],
              language='c++'),
]

setup(name='hello-world-package',
      version='1.0',
      description='hello world in pybind11',
      author='fangjun',
      author_email='fangjun dot kuang at gmail dot com',
      url='https://github.com/csu-fangjun',
      long_description='This is really just a demo package.',
      ext_modules=ext_modules)
