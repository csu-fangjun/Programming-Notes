
x86 asm
========


x86 asm on 64-bit
-----------------

To setup the environment for compiling x86 asm on 64-bit Ubuntu:
(refer to `<https://denniskubes.com/2017/01/31/compiling-x86-assembly-on-x64-linux/>`_)

.. code-block::

   sudo dpkg --add-architecture i386
   sudo apt-get update
   sudo apt-get dist-upgrade
   sudo apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386
   sudo apt-get install multiarch-support
   sudo apt-get install gcc-multilib g++-multilib


.. code-block::

   gcc -m32 hello_world.s

.. literalinclude:: ./code/x86_asm/hello_world_x64.s
  :caption: hello_world_x64.s
  :language: asm
  :linenos:

.. literalinclude:: ./code/x86_asm/hello_main.s
  :caption: hello_main.s
  :language: asm
  :linenos:

.. literalinclude:: ./code/x86_asm/hello_world.s
  :caption: hello_world.s
  :language: asm
  :linenos:

.. NOTE::

  For ``as`` and ``ld``, the entry point is ``_start`` and we have to call
  ``exit`` ourselves.

  For ``gcc``, the entry point is ``main``.



datatype
--------

.. literalinclude:: ./code/x86_asm/datatype.c
  :caption: datatype.c
  :language: c
  :linenos:

.. literalinclude:: ./code/x86_asm/datatype.s
  :caption: datatype.s
  :language: asm
  :linenos:

.. literalinclude:: ./code/x86_asm/Makefile
  :caption: Makefile
  :language: makefile
  :linenos:


References
----------

- `<http://flint.cs.yale.edu/cs421/papers/x86-asm/asm.html>`_
- x64 Cheat Sheet

   `<https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf>`_
