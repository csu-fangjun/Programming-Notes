Functions
=========

Args
----

Note:

  - How can a function accept an ``*arg``? How to use it inside the function?
  - How can a function accept an ``**kwarg``? How to use it inside the function?

``*arg`` can be considered as varidic arguments in C++. Inside the function,
``*arg`` unpacks the arguments. ``arg`` is a tuple inside the function.
Use ``*arg`` to pass it to another function.

.. literalinclude:: ./code/functions_test/args_test.py
  :caption: code/functions_test/args_test
  :language: python
  :linenos:
