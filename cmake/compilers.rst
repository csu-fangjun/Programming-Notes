
Compilers
=========

To use a customized gcc:

.. code-block:: bash

  export CC=/path/to/gcc-x.x.x/bin/gcc
  export CXX=/path/to/gcc-x.x.x/bin/g++
  export LIBRARY_PATH=/path/to/gcc-x.x.0/lib:/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
  export LD_LIBRARY_PATH=/path/to/gcc-x.x.x/lib64:$LD_LIBRARY_PATH
  export C_INCLUDE_PATH=/path/to/gcc-x.x.x/include
  export CPLUS_INCLUDE_PATH=/path/to/gcc-x.x.x/include
  cmake -DCMAKE_BUILD_TYPE=Debug ..
  make -j

