
Coverage
========

Refer to `<https://coverage.readthedocs.io/en/v4.5.x/install.html>`_.

Install::

  pip install coverage
  pip install pytest

Sample config file ``.coveragerc``::

  [run]
  branch = True
  include = /path/to/dir/*.py

Usage::

  coverage run -m pytest /path/to/dir
  coverage report
  # or use html
  coverage html # generate the result inside the dir `htmlconv`

pytest
------

.. code-block::

  pytest . # use all *_test.py
  pytest xxx_test.py
  pytest xxx_test.py -s   # enable print to console
