
Magic Methods
=============

Example 1
---------

.. literalinclude:: ./code/magic_methods_test/ex1.py
  :caption: code/magic_methods_test/ex1.py
  :language: python
  :linenos:

Example 2
---------

``__new__`` and ``__init__``.

See:

  - `<https://docs.python.org/3/reference/datamodel.html?highlight=__new__#object.__new__>`_
  - `<https://docs.python.org/3/reference/datamodel.html?highlight=__new__#object.__init__>`_


.. literalinclude:: ./code/magic_methods_test/ex1.py
  :caption: code/magic_methods_test/ex1.py
  :language: python
  :linenos:

Output::

  called Foo __new__: args = (), kwargs = {}
  140462185990464
  In init: 140462185990464
  f1: 140462185990464
  called Foo __new__: args = (1, 2), kwargs = {'a': 3, 'b': 4}
  140462185809376
  In init: 140462185809376
  f2: 140462185809376
