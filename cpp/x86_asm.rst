
X86 ASM
=======

- ``gcc -S add.c``
- ``gcc -S -o add_no_cfi.s -fno-asynchronous-unwind-tables  add.c``

.. literalinclude:: ./code/asm/add.c
  :caption: add.c
  :language: c
  :linenos:

.. literalinclude:: ./code/asm/add.s
  :caption: add.s
  :language: asm
  :linenos:

If ``int main()`` invokes ``add``, ``CFA`` is
``sp`` in ``main`` before calling ``add``, after calling ``add``,
it pushes the return address onto the stack; hence:

.. code-block::

  sp = sp - 8
  *sp = return address

Inside ``add``, it uses ``pushq %rbp``, which translates to:

.. code-block::

  sp = sp - 8
  *sp = rbp

Now we have ``sp == CFA -  16``, so it uses::

  .cfi_def_cfa_offset 16

which means ``CFA == sp + 16``.


``.cfi_offset 6, -16`` means the 6-th register is saved at
offset ``-16`` from ``CFA``. The 6-th register is ``rbp``.

``.cfi_def_cfa_register 6`` means it will use the 6-th register
as ``CFA``, so now ``CFA == rbp``.

``.cfi_def_cfa 7, 8`` means to take the 7-th register and add offset 8
to it to get the ``CFA``. The 7-th register is ``sp``, so ``CFA = sp + 8``.

.. literalinclude:: ./code/asm/add_no_cfi.s
  :caption: add_no_cfi.s
  :language: asm
  :linenos:

cfi
---

cfi is short for call frame information.

CFA is short for Canonical Frame Address.

Refer to
- `<https://sourceware.org/binutils/docs-2.24/as/CFI-directives.html#CFI-directives>`_.
- CFI support for GNU assembler (GAS) `<http://www.logix.cz/michal/devel/gas-cfi/>`_.

