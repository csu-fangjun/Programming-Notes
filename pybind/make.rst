
Build with Make
===============

``pybind11`` is a header only library. To let the compiler
know the places where pybind11 header files are, use

.. code-block:: bash

  python3 -m pybind11 --includes

It should print something like::

  -I/path/to/py35/include/python3.5m -I/path/to/py35/include/site/python3.5


In addition, the generated library has a special suffix
for the filename. Something like ``.cpython-35m-x86_64-linux-gnu.so``,
which can be obtained by

.. code-block:: bash

  python3-config --extension-suffix

To compile the following code,

.. literalinclude:: ./code/hello.cc
  :caption: hello.cc
  :language: cpp
  :linenos:

We can use

.. literalinclude:: ./code/Makefile
  :caption: Makefile
  :language: makefile
  :linenos:

``make all`` will generate a file
whose name looks like ``hello.cpython-35m-x86_64-linux-gnu.so``.

To use it in ``Python``, modify ``PYTHONPATH`` and use ``import hello``
to import this module.

