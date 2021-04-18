iterator
========

See

  - PEP 234 -- Iterators

    `<https://www.python.org/dev/peps/pep-0234/>`_

    It says ``for key in dict_obj`` is more efficient than ``for key in dict_obj.keys()``.

.. literalinclude:: ./code/iterator_test/ex1.py
  :caption: code/iterator_test/ex1.py
  :language: python
  :linenos:


See `PEP 3114 -- Renaming iterator.next() to iterator.__next__() <https://www.python.org/dev/peps/pep-3114/>`_
for ``next()`` and ``__next__()``.
