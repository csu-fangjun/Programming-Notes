
The ``with`` Statement
======================

contextlib.contextmanager
-------------------------

See `<https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager>`_

An example from PyTorch

.. code-block:: python

  import contextlib
  @contextlib.contextmanager
  def device(idx):
      prev_idx = torch._C._cuda_getDevice()
      torch._C._cuda_setDevice(idx)
      yield
      torch._C._cuda_setDevice(prev_idx)

  @contextlib.contextmanager
  def _dummy_ctx():
      yield

.. literalinclude:: ./code/generator_test.py
  :caption: generator_test.py
  :language: python
  :linenos:


References
----------

- `PEP 343 -- The "with" Statement`_
- `<https://docs.python.org/3/reference/compound_stmts.html#the-with-statement>`_
- `<https://docs.python.org/3/reference/datamodel.html#with-statement-context-managers>`_

.. _PEP 343 -- The "with" Statement: https://www.python.org/dev/peps/pep-0343/
