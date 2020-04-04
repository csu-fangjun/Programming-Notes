
Arm Compute Library
===================

Its C++ code documentation is really nice!

Cross Tool Chain
----------------

Refer to `<https://releases.linaro.org/components/toolchain/binaries/>`_.

For example:

.. code-block:: bash

  cd ~/software
  wget https://releases.linaro.org/components/toolchain/binaries/4.9-2016.02/aarch64-linux-gnu/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu.tar.xz
  export PATH=$HOME/software/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu/bin:$PATH

``aarch64-linux-gnu-gcc --version`` should print::

    aarch64-linux-gnu-gcc (Linaro GCC 4.9-2016.02) 4.9.4 20151028 (prerelease)
    Copyright (C) 2015 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

For ``armv7a``, install:

.. code-block:: bash

  wget https://releases.linaro.org/components/toolchain/binaries/4.9-2016.02/arm-linux-gnueabihf/gcc-linaro-4.9-2016.02-x86_64_arm-linux-gnueabihf.tar.xz

.. NOTE::

  For 32-bit amr (armv7a), there is ``arm-linux-gnueabi`` and ``arm-linux-gnueabihf``.
  For ``arm-linux-gnueabi``, it has default option ``-mfloat-abi=softfp``;
  For ``arm-linux-gnueabihf``, it has default option ``-mfloat-abi=hard``;
  Both ``softfp`` and ``hard`` requires hardward support. The third alternative
  ``-mfloat-abi=soft`` is pure software.

Compilation
-----------

.. code-block:: bash

  pip install scons
  cd ComputeLibrary

  scons Werror=1 -j8 debug=1 neon=1 opencl=0 os=linux arch=arm64-v8a build_dir=./build

.. HINT::

  The default toochain prefix for ``arm64-v8a`` is ``aarch64-linux-gnu-``.
  If we have have a toolchain ``aarch64-foobar-linux-gcc``, then
  we can set ``toolchain_prefix=aarch64-foobar-linux-``. Refer to
  the script ``SConstruct``.

After compilation, we can find the following files::

  ls build/build/lib*

  build/build/libarm_compute_core.so (58 MB)
  build/build/libarm_compute_core-static.a (161 MB)
  build/build/libarm_compute_graph.so (21 MB)
  build/build/libarm_compute_graph-static.a (63 MB)
  build/build/libarm_compute.so (32 MB)
  build/build/libarm_compute-static.a (89 MB)

Hello World
-----------

The latest version of arm compute library is ``v20.02.1``.

qemu
::::

First, install it::

  sudo apt-get install qemu

Then run it::

  qemu-aarch64 ./hello

It will print::

  /lib/ld-linux-aarch64.so.1: no such file or directory

To figure out the solution, we have to find the missing file via::

  find /path/to/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu -name ld-linux*

which prints::

  /path/to/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/lib/ld-linux-aarch64.so.1

So, we can use::

  qemu-aarch64 -L /path/to/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc ./hello
