
LLVM
====

API Doc: `<https://llvm.org/doxygen/>`_.

References
----------

- Getting started with LLVM

    `<http://www.cs.utexas.edu/~pingali/CS380C/2019/assignments/llvm-guide.html>`_

TODO
----

- ``SmallVector<int, 4>``
- ``SmallVectorImpl``
- ``StringRef``
- ``StringLiteral``
- ``Twine``
- ``raw_ostream``
- ``ArrayRef<int*>``

C++ABI
------

.. code-block::

  cd llvm-project
  mkdir build-clang
  cd build-clang
  cmake \
    -G Ninja \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DCMAKE_INSTALL_PREFIX="$HOME/software/llvm-project/clang" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=On \
    ..
  ninja -j10
  ninja install

Note that add ``-DLLVM_TARGETS_TO_BUILD=host`` will reduce the building time.
Its default value is ``all``, meaning ``all`` platforms.

After building, the binaries are in ``$build_dir/bin``. Add it to ``PATH``.

