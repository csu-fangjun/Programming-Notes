Intermediate representation
===========================

.. code-block:: bash

    clang --help

      -emit-llvm  Use the LLVM representation for assembler and object files
      -S          Only run preprocess and compilation steps

- ``clang -S -emit-llvm ex.c`` generates a text file ``ex.ll``.
- ``clang -c -emit-llvm ex.c`` generates a binary file ``ex.bc``.
- ``llvm-dis ex.bc`` generates a file ``ex.ll``, which is identical with the file
  generated using ``clang -S -emit-llvm ex.c`.
- ``llvm-as ex.ll`` generates a file ``ex.bc``, which is identical with the file
  generated using ``clang -c -emit-llvm ex.c``.
- ``llc ex.ll`` generates the assembly file ``ex.s``

