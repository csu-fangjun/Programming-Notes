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
- ``lli ex.ll`` can run this file. Use ``echo $?`` to see the return value.


See `<https://llvm.org/devmtg/2019-04/slides/Tutorial-Bridgers-LLVM_IR_tutorial.pdf>`_.
