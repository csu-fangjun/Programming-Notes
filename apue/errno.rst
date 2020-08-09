
errno
=====

It is thread safe.

.. code-block::

  sudo apt-get install moreutils

.. code-block::

  man errno       # for command line tool
  man 3 errno     # for API


- List all errno and its descriptions:

.. code-block::

  $ errno -l

  EPERM 1 Operation not permitted
  ENOENT 2 No such file or directory
  ESRCH 3 No such process
  EINTR 4 Interrupted system call

- List a specified errno

.. code-block::

  $ errno  1

  EPERM 1 Operation not permitted

  $ errno EPERM

  EPERM 1 Operation not permitted

- Search for a specific string; it is case-insensitive!

.. code-block::

  $ errno -s Operation

  EPERM 1 Operation not permitted
  ENOTSOCK 88 Socket operation on non-socket
  EOPNOTSUPP 95 Operation not supported
  EALREADY 114 Operation already in progress
  EINPROGRESS 115 Operation now in progress
  ECANCELED 125 Operation canceled
  ERFKILL 132 Operation not possible due to RF-kill
  ENOTSUP 95 Operation not supported

Examples:


.. literalinclude:: ./code/errno-test.c
  :caption: errno-test.c
  :language: cpp
  :linenos:
