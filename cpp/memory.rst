
Memory
======

glibc malloc
------------

.. code-block::

  wget http://mirrors.ustc.edu.cn/gnu/libc/glibc-2.20.tar.bz2
  tar xf glibc-2.20.tar.bz2

There are various implemenations of libc, see
`<https://en.wikipedia.org/wiki/C_standard_library>`_.



A naive malloc
--------------

Refer to `<https://danluu.com/malloc-tutorial/>`_.

.. literalinclude:: ./code/memory/malloc.cc
  :caption: A naive implementation of malloc
  :language: cpp
  :linenos:



TODO
----

- dlmalloc – General purpose allocator
- ptmalloc2 – glibc
- jemalloc – FreeBSD and Firefox
- tcmalloc – Google
- libumem – Solaris

References
----------

- How do malloc() and free() work?

    `<https://stackoverflow.com/questions/1119134/how-do-malloc-and-free-work>`_

- `<https://github.com/OpenSIPS/opensips/tree/master/mem>`_


- Dynamic Storage Allocation: A Survey and Critical Review'

    A paper: `<https://users.cs.northwestern.edu/~pdinda/icsclass/doc/dsa.pdf>`_

- GO MEMORY MANAGEMENT


    `<https://povilasv.me/go-memory-management/>`_

- Operating Systems Development - Virtual Memory

    `<http://brokenthorn.com/Resources/OSDev18.html>`_

- Operating System Development Series

    `<http://brokenthorn.com/Resources/OSDevIndex.html>`_

- Writing a Simple Operating System — from Scratch

    `<https://www.cs.bham.ac.uk/~exr/lectures/opsys/10_11/lectures/os-dev.pdf>`_
