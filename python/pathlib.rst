pathlib
=======

See

  - `<https://docs.python.org/3/library/pathlib.html>`_
  - PEP 428 -- The pathlib module -- object-oriented filesystem paths

    `<https://www.python.org/dev/peps/pep-0428/>`_

    It was proposed in 2012.

Its implementation is in `<https://github.com/python/cpython/blob/master/Lib/pathlib.py>`_.

Note for reading the code:

  1. Define global tuple variables and access it in functions. See
       `<https://github.com/python/cpython/blob/adf24bd835ed8f76dcc51aa98c8c54275e86965b/Lib/pathlib.py#L34>`_

Example 1
---------

.. literalinclude:: ./code/pathlib_test/ex1.py
  :caption: code/pathlib_test/ex1.py
  :language: python
  :linenos:
