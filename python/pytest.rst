pytest
======

We need to install it first:

.. code-block:: bash

   pip install pytest


``-s`` to also print to the console.

``-q`` to be quiet in the output.

See `<https://docs.pytest.org/en/6.2.x/index.html>`_

pytest configuration in PyTorch: `<https://github.com/pytorch/pytorch/blob/master/pytest.ini>`_,
which is added in `<https://github.com/pytorch/pytorch/pull/53152>`_

By default, pytest will run all ``test_*.py`` and ``*_test.py`` files
in the current directory and its subdirectories.

We can use ``testpaths`` `<https://docs.pytest.org/en/6.2.x/reference.html#confval-testpaths>`_
to specify the test directory to be searched.

If the name of a class begins with `Test`, then all methods whose names begins with
``test_`` is also a candiate to be run.

.. code-block::

   pytest -x # stop after first failure, -x is equivalent to -exitfirst
   pytest --maxfail=2  # stop after two failures

   pytest -s # show output from `print`.

   pytest test_mod.py
   pytest ./abc
   pytest -k "MyClass and not method" # Matches TestMyClass.test_something, but not MyClass.test_method_simple

   pytest test_mod.py::test_func
   pytest test_mod.py::TestClass::test_method

   pytest -m slow # run all tests decorated by @pytest.mark.slow

   pytest --pyargs pkg.testing # import pkg.testing and find tests in it and run

   pytest --pdb # run pdb on failure
   pytest -x --pdf # run pdb on the first failure
   pytest --maxfail=3 --pdb # run pdf on the first 3 failures
   pytest --trace # run pdb at the start of a test

   # set a breakpoint
   import pdb; pdb.set_trace()


.. literalinclude:: ./code/pytest_test/test_ex1.py
  :caption: code/pytest_test/test_ex1.py
  :language: python
  :linenos:

.. literalinclude:: ./code/pytest_test/test_ex2.py
  :caption: code/pytest_test/test_ex2.py
  :language: python
  :linenos:

.. literalinclude:: ./code/pytest_test/test_ex3.py
  :caption: code/pytest_test/test_ex3.py
  :language: python
  :linenos:
