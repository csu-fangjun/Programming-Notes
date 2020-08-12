
System Programming
==================

parameter passing from c/c++ to assembly
----------------------------------------

For x86_64,

.. code-block::

  void parameters(rdi, rsi, rdx, rcx, r8, r9) {}

  // rbp -> sp
  // rbp + 8 -> return address
  // rbp + 16 -> the 7-th argument

For x86, parameters are passed via stack.

.. code-blck::

  // ebp -> sp
  // ebp + 4 -> return address
  // ebp + 8 -> 0th argument
  // ebp + 12 -> 1st argument


syscall
-------

**parameter passing for x86:**

- if number of parameters is <= 5, then it is via ``ebx``, ``ecx``, ``edx``, ``esi`` and ``edi``
- if number of parameters is >5, then it is via ``ebx``. Parameters are saved in a user buffer
  and the address is passed to the kernel.

**file descriptor**

- ``STDIN_FILENO``, ``STDOUT_FILENO``, ``STDERR_FILENO``


References
----------

- Linux System Programming

    A book written by Robert Love.
