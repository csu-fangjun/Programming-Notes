File I/O
========

umask
-----

.. code-block::

  help umask # a builtin bash command
  man umask  # a system call

Default value of umask:

.. code-block::

  $ umask
  0002

  $ umask -p
  umask 0002

  $ umask -S
  u=rwx,g=rwx,o=rx


When umask is ``0002``, the permission of the file ``a`` created by ``touch a``
is ``-rw-rw-r--``.

When umask is ``0000``, the permission of the file ``a`` is ``-rw-rw-rw-``
and ``umask -S`` outputs ``u=rwx,g=rwx,o=rwx``.

System Calls
------------

.. code-block::

  man stdin

It contains help information for ``STDIN_FILENO``, ``STDOUT_FILENO``,
and ``STDERR_FILENO``; they are defined in ``<unistd.h>``.
