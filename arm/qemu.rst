
Qemu
====

Install from ``apt-get``
------------------------

If you have ``sudo`` permission, the step to install ``qemu`` is very simple::

  sudo apt-get install qemu

Install from source
-------------------

To install it from source, use the following commands [1]_::

  cd $HOME/open-source
  git clone --depth 1 git://git.qemu-project.org/qemu.git
  cd qemu
  ../configure --help
  ../configure \
    --prefix=$HOME/software/qemu \
    --target-list=aarch64-softmmu,arm-softmmu,aarch64-linux-user,arm-linux-user

It will display the following message::

  ERROR: glib-2.48 gthread-2.0 is required to compile QEMU

To fix it, install the following library::

  sudo apt-get install libglib2.0-dev

When the above ``./configure`` command is executed again, it shows::

  ERROR: pixman >= 0.21.8 not present.
         Please install the pixman devel package.

We have to install ``pixman``::

  sudo apt-get install libpixman-1-dev

Rerun the above ``configure`` commands::

  ../configure \
    --prefix=$HOME/software/qemu \
    --target-list=aarch64-softmmu,arm-softmmu,aarch64-linux-user,arm-linux-user
  make -j20
  make install

After installation, we can find the following binaries in ``$HOME/software/qemu/bin``::

  elf2dmp         ivshmem-server  qemu-arm   qemu-ga   qemu-io   qemu-pr-helper       qemu-system-aarch64
  ivshmem-client  qemu-aarch64    qemu-edid  qemu-img  qemu-nbd  qemu-storage-daemon  qemu-system-arm

.. [1] `Building from source <https://en.wikibooks.org/wiki/QEMU/Installing_QEMU>`_

Hello Qemu
----------

To test ``qemu``, write a simple hello world program:

.. code-block:: cpp

  #include <iostream>

  int main() {
    std::cout << "hello qemu" << std::endl;
    return 0;
  }

Then compile it::

  aarch64-linux-gnu-g++ -o hello hello.cc

Run it::

  qemu-aarch64 ./hello

which shows the following error::

  /lib/ld-linux-aarch64.so.1: No such file or directory

We can find ``ld-linux-aarch64.so.1`` in the cross toolchain folder:

.. code-block:: console

  $ find /path/to/software/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu/ -name "ld-linux*"
  /path/to/software/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/lib/ld-linux-aarch64.so.1

So we have to invoke ``qemu`` like this::

  $ qemu-aarch64 -L /path/to/software/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc ./hello

And it displays the following message::

  hello qemu

Now we are ready to compile and run our ARM programs.
