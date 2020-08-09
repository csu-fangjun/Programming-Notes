
mypy
====

mypy requires python 3.5 or greater.

.. code-block::

  pip install mypy
  pip install mypy-extensions


Open Source
------------

- PyTorch

    `<https://github.com/pytorch/pytorch/blob/master/mypy.ini>`_

- PyTorch Audi

    `<https://github.com/pytorch/audio/blob/master/mypy.ini>`_


Sample config file ``mypy.ini``::

  [mypy]
  files = /path/to/dir

  [mypy-some_third_party_module]
  ignore_missing_imports = True


References
----------

- Type hints cheat sheet (Python 3)


    `<https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html#cheat-sheet-py3>`_

- PEP 3107 -- Function Annotations (2006)

    `<https://www.python.org/dev/peps/pep-3107/>`_

    It defines the syntax for function annotations.

    Example:

    .. code-block::

      def hello(s: str) -> None:
        return s

- PEP 484 -- Type Hints (2014)

    `<https://www.python.org/dev/peps/pep-0484/>`_

- https://www.python.org/dev/peps/pep-0483/#fundamental-building-blocks

    It describes ``List``, ``Tuple``, ``Union``, ``Optional``, ``Iterable``,
    ``Any``, ``Callable``.


- PEP 526 -- Syntax for Variable Annotations

    `<https://www.python.org/dev/peps/pep-0526/>`_

    Annotations for variables.


