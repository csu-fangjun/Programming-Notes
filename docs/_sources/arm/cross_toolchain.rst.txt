Cross Tool Chain
================

We have to install a compiler to compile code for ARM.
One possible toolchain is from Linaro:

.. code-block::

  cd ~/software
  wget https://releases.linaro.org/components/toolchain/binaries/4.9-2016.02/aarch64-linux-gnu/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu.tar.xz
  tar xf gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu.tar.xz
  export PATH=$HOME/software/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu/bin:$PATH


The following toolchain can be used for armv7a:

.. code-block:: bash

  wget https://releases.linaro.org/components/toolchain/binaries/4.9-2016.02/arm-linux-gnueabihf/gcc-linaro-4.9-2016.02-x86_64_arm-linux-gnueabihf.tar.xz

.. NOTE::

  There are ``arm-linux-gnueabi`` and ``arm-linux-gnueabihf``
  for armv7a. What are the differences between them?

  - ``-mfloat-abi=softfp`` is default in ``arm-linux-gnueabi``
  - ``-mfloat-abi=hard`` is default in ``arm-linux-gnueabihf``

After a successful installation, modify ``PATH`` and run the following command:

.. code-block:: console

  $ aarch64-linux-gnu-gcc --version
  aarch64-linux-gnu-gcc (Linaro GCC 4.9-2016.02) 4.9.4 20151028 (prerelease)
  Copyright (C) 2015 Free Software Foundation, Inc.
  This is free software; see the source for copying conditions.  There is NO
  warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Now we are ready to dive into ARM programming.
