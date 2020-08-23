
Call convention
===============

x86_64
------

.. code-block::

  void test(rdi, rsi, rdx, rcx, r8, r9);
             1    2    3    4    5   6

Extra parameters are pass via stack.

Refer to:

- `<https://aaronbloomfield.github.io/pdr/book/x86-64bit-ccc-chapter.pdf>`_

    The 64 bit x86 C Calling Convention
