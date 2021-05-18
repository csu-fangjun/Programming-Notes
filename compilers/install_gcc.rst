Install gcc
===========

Download gcc source code and unzip

.. code-block::

  sudo apt-get install libgmp-dev libmpfr-dev libmpc-dev libc6-dev
  sudo apt-get install gcc-multilib # for compiling 32-bit programs on 64-bit machines
  mkdir build-gcc-4.8.5
  cd build-gcc-4.8.5
  ../gcc-4.8.5/configure --prefix=/path/to/install/gcc-4.8.5
  make -j

For gcc-4.8.5, to solve the error related to `libc_name_p`,
download this patch and save it as `a.txt`:

.. code-block::

  cd gcc-4.8.5
  patch -p1 < a.txt
