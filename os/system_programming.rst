
System Programming
==================

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
