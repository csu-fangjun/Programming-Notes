subprocess
==========

Note `subprocess.run()`  poses a constraint that
only keyword arguments are supported after ``args``.

Note that the following functions are deprecated and can be replaced
by ``subprocess.run()``:

  - ``subprocess.call()``
  - ``subprocess.check_call()``
  - ``subprocess.check_output()``


Example 1
---------

.. literalinclude:: ./code/subprocess_test/ex1.py
  :caption: code/subprocess_test/ex1.py
  :language: python
  :linenos:

subprocess has attributes:

  - ``subprocess.DEVNULL``
  - ``subprocess.PIPE``
  - ``subprocess.STDOUT``, this is usally assigned to the input argument ``stderr``.
