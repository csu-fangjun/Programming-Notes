fd_set
======

.. code-block::

  man select

  FD_ZERO
  FD_SET
  FD_ISSET
  FD_CLR

It is useful in synchronous I/O.

Example 1
---------

.. literalinclude:: ./code/fd-set-test.c
  :caption: code/fd-set-test.c
  :language: c
  :linenos:

