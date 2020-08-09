
Install pybind11
================

The installation is very simple:

.. code-block:: bash

    pip install pybind11

To check that ``pybind11`` is installed successfully, run

.. code-block:: bash

    python3 -m pybind11 --includes

It should print something like::

  -I/path/to/py35/include/python3.5m -I/path/to/py35/include/site/python3.5

``/path/to/py35/include/python3.5m`` contains header files for ``Python``;
while ``/path/to/py35/include/site/python3.5`` contains header files
for ``pybind11``.
