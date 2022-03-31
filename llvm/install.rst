Installation
============

Refer to `<https://llvm.org/docs/GettingStarted.html#requirements>`_

.. cod-block:: bash

    git clone --depth 1 https://github.com/llvm/llvm-project.git
    cd llvm-project
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -G Ninja ../llvm
    # cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld" -G Ninja ../llvm

    cmake --build . --target <target>
    # or
    export PATH=$PWD/build/bin:$PATH
    export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH

`<https://llvm.org/docs/GettingStarted.html#getting-a-modern-host-c-toolchain>`_
describes how to install GCC from source.

Use of a user provided GCC:

.. code-block:: bash

  mkdir build
  cd build
  CC=$HOME/toolchains/bin/gcc CXX=$HOME/toolchains/bin/g++ \
    cmake .. -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath,$HOME/toolchains/lib64 -L$HOME/toolchains/lib64"

Useful tools that can be found in `build/bin`:

.. code-block::

  $ llvm-config --cxxflags
  -I/ceph-fj/fangjun/open-source-2/llvm-project/llvm/include -I/ceph-fj/fangjun/open-source-2/llvm-project/build/include -std=c++14   -fno-exceptions -fno-rtti -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS

  $ llvm-config --libdir
  /ceph-fj/fangjun/open-source-2/llvm-project/build/lib

  $ llvm-config --cflags
  -I/ceph-fj/fangjun/open-source-2/llvm-project/llvm/include -I/ceph-fj/fangjun/open-source-2/llvm-project/build/include  -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS

  $ llvm-config --ldflags
  -L/ceph-fj/fangjun/open-source-2/llvm-project/build/lib

  $ llvm-config --src-root
  /ceph-fj/fangjun/open-source-2/llvm-project/llvm

  $ llvm-config --obj-root
  /ceph-fj/fangjun/open-source-2/llvm-project/build

  $ llvm-config --version
  15.0.0git

  $ llvm-config --bindir
  /ceph-fj/fangjun/open-source-2/llvm-project/build/bin
