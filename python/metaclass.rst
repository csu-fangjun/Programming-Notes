metaclass
=========

.. code-block:: python3

  print(help(type))

shows three usages of ``type``:

.. code-block::

  class type(object)
   |  type(object_or_name, bases, dict)
   |  type(object) -> the object's type
   |  type(name, bases, dict) -> a new type

See `<https://stackoverflow.com/questions/12971641/need-to-understand-the-flow-of-init-new-and-call>`_
for a description of:

  - ``__call__``
  - ``__new__``
  - ``__init__``

See:

  - `<https://docs.python.org/3/library/functions.html#type>`_

      The official documentation for ``type()``.

  - `<https://www.python.org/dev/peps/pep-3115/>`_

  - `<https://eli.thegreenplace.net/2012/04/16/python-object-creation-sequence>`_

      A very good introduction to ``__call__``, ``__new__``, ``__init__``.
      It also describes the underlying C implementation in CPython.

  - `<https://eli.thegreenplace.net/2011/08/14/python-metaclasses-by-example/>`_

      A blog artical explanins how metaclass works in Python.
