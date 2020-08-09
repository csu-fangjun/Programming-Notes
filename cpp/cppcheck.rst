
cppcheck
========

Install from source
-------------------

.. code-block::

  git clone --depth 1 https://github.com/danmar/cppcheck.git
  cd cppcheck
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=$HOME/software/cppcheck

suppression file
----------------

**Example**
  `<https://gitlab.kitware.com/vtk/vtk/commit/13f8a0d4f7a0753aec5d4b6e9d26db647c10786d>`

Useful options:
  - ``-q``
  - ``--template='{file}:{line},{severity},{id},{message}'``
