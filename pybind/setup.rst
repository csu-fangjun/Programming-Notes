
Build with setup.py
===================

To compile the following code,

.. literalinclude:: ./code/hello.cc
  :caption: hello.cc
  :language: cpp
  :linenos:

We can use

.. literalinclude:: ./code/setup.py
  :caption: setup.py
  :language: python
  :linenos:

Execute ``python setup.py build`` and it will generate
``build/lib.linux-x86_64-3.5/hello.cpython-35m-x86_64-linux-gnu.so``.


.. code-block:: bash

    >>> import sys
    >>> sys.path.insert(0, 'build/lib.linux-x86_64-3.5')
    >>> import hello
    >>> hello.add(1, 2)
    3
