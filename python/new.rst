new
===

``__new__``.

**Question 1**: What arguments do ``__new__`` accept?

``__new__`` is a classmethod, whether you add ``@classmethod`` or not.
So the first argument of ``__new__`` is a type of the current class.
The argument name is usually ``cls``.

The remaining arguments of ``__new__`` depend on how you invoke the class'
constructor, but you can always use ``__new__(cls, *args, **kwargs)``.

``*args`` and ``**kwargs`` are passed to ``__init__``.

Example 1
---------

.. literalinclude:: ./code/new_test/ex1.py
  :caption: code/new_test/ex1.py
  :language: python
  :linenos:
