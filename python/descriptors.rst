Descriptors
===========

Goals: To know the internals of the following:

  - How is ``classmethod`` implemented
  - How is ``staticmethod`` implemented
  - How is ``property`` implemented

  - Differences between `class.__dict__` and `instance.__dict__`.



.. literalinclude:: ./code/descriptor_test.py
  :caption: code/descriptor_test.py
  :language: python
  :linenos:


- ``object.__set_name__(self, owner, name)``: `<https://docs.python.org/3/reference/datamodel.html#object.__set_name__>`_

- ``object.__get__(self, instance, owner=None)``: `<https://docs.python.org/3/reference/datamodel.html#object.__get__>`_

  - ``__get__`` is often used for methods, e.g., static methods and class methods.

  - A descriptor with only ``__get__`` defined is called a non-data descriptor.

- ``object.__set__(self, instance, value)``: `<https://docs.python.org/3/reference/datamodel.html#object.__set__>`_

    - To define a read only property, throw ``AttributeError`` in ``__set__``.



References
----------

- Descriptor HowTo Guide

    `<https://docs.python.org/3/howto/descriptor.html>`_

- PEP 252 -- Making Types Look More Like Classes

    `<https://www.python.org/dev/peps/pep-0252/>`_

- 3.3.2.2. Implementing Descriptors

    `<https://docs.python.org/3/reference/datamodel.html#descriptors>`_

