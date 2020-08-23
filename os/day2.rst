
Day 2
=====

.. HINT::

  ``gcc hello.S`` invokes the macro processing step.

Exercise 1
----------

Source code
^^^^^^^^^^^

.. literalinclude:: ./code/day2/ex1/hello.S
  :caption: hello.s
  :lineno-start: 2
  :lines: 2-45
  :linenos:

Compile and link
^^^^^^^^^^^^^^^^^

.. code-block::

  gcc -m32 -c -o hello.o hello.S
  ld -m elf_i386 -o hello hello.o

Section header
^^^^^^^^^^^^^^

Every section has a number. For example, ``.text`` has number 1.

The symbol table has a column ``Ndx`` indicating to which section
the symbol belongs. For example, ``s0`` has number ``3``, which is the section
``.bss``; ``bar`` has number ``1``, which is the section ``.text``.

.. literalinclude:: ./code/day2/ex1/hello.S
  :caption: section header of hello.o
  :lineno-start: 51
  :lines: 51-68
  :linenos:

.. literalinclude:: ./code/day2/ex1/hello.S
  :caption: section header of hello
  :lineno-start: 123
  :lines: 123-139
  :linenos:

Symbol table
^^^^^^^^^^^^

- The first column ``number`` of a symbol is used in the relocation symbol.
- The second column ``value``:
  for ``hello.o``, it's the byte offset since every sections starts from address 0;
  for ``hello``, it's the absolute address of the symbol. (TODO: this is not correct
  for ``COMM`` symbols.)
- The first column of ``nm`` is the same as the second column of ``readelf -s``.

.. literalinclude:: ./code/day2/ex1/hello.S
  :caption: symbol table of hello.o
  :lineno-start: 70
  :lines: 70-88
  :linenos:

.. literalinclude:: ./code/day2/ex1/hello.S
  :caption: relocation table of hello.o
  :lineno-start: 141
  :lines: 141-164
  :linenos:

.. literalinclude:: ./code/day2/ex1/hello.S
  :caption: symbol table of hello.o
  :lineno-start: 192
  :lines: 192-204
  :linenos:

.. literalinclude:: ./code/day2/ex1/hello.S
  :caption: symbol table of hello.o
  :lineno-start: 206
  :lines: 206-221
  :linenos:

Disassemble
^^^^^^^^^^^

.. literalinclude:: ./code/day2/ex1/hello.S
  :caption: disassemble of hello.o
  :lineno-start: 90
  :lines: 90-108
  :linenos:

.. literalinclude:: ./code/day2/ex1/hello.S
  :caption: disassemble of hello
  :lineno-start: 166
  :lines: 166-186
  :linenos:

Relocation table
^^^^^^^^^^^^^^^^

``.rel.text`` is for ``.text``. ``.rel.xxx`` is for ``.xxx``.

For
.. code-block::

     Offset     Info    Type            Sym.Value  Sym. Name
     00000002  00000b01 R_386_32          00000008   c1

Offset ``0002`` means the position for the relocation in ``.text`` is at byte offset 2, which is for

.. code-block::

    00000000 <_start>:
       0: 8d 05 00 00 00 00     lea    0x0,%eax

The Info ``000b01`` has two parts: ``0b`` is the symbol number of the relocation symbol
in the symbol table, which is

.. code-block::

    11: 00000008     8 OBJECT  GLOBAL DEFAULT  COM c1

.. NOTE::

  The symbol table uses decimal, which the relocation table uses hexdecimal.

The ``01`` part in ``0b01`` is the numerical representation of the type of the relocation,
which is further described by the type ``R_386_32``.

For

.. code-block::

  00000027  00000e02 R_386_PC32        00000017   foo

``0027`` is the byte offset in ``.text`` that requires relocation, which is

.. code-block::

    00000026 <bar>:
      26: e8 fc ff ff ff        call   27 <bar+0x1>

``0e`` in ``0e02`` is the symbol number in the symbol table, which is
.. code-block::

  14: 00000017     0 NOTYPE  GLOBAL DEFAULT    1 foo

``02`` in ``0e02`` is the type of the relocation, which is also described
by ``R_386_PC32``.

``0017`` is the value of ``foo``, which is the byte offset of ``foo` in ``.text``
and is shown below

.. code-block::

    00000017 <foo>:
      17: bb 00 00 00 00        mov    $0x0,%ebx
      1c: be 04 00 00 00        mov    $0x4,%esi
      21: bf 07 00 00 00        mov    $0x7,%edi

For

.. code-block::

    00000026 <bar>:
      26: e8 fc ff ff ff        call   27 <bar+0x1>

``e8`` is the op code for ``call``. The remaining 4 bytes ``fc ff ff ff``
is ``-4``. After relocation, the address of the function to be called is

.. code-block::

    func_address = PC + xxx = relocation_offset + 4 + xxx = relocation_offset - 0xfffffc + xxx

so ``xxx`` is

.. code-block::

  xxx = func_address - PC = func_address + 0xfffffc - relocation_offset

For ``hello.o``, ``func_address`` is the symbol value of ``foo``, which is ``0x17``;
``relocation_offset`` is ``0x27``. So

.. code-block::

  xxx = 0x17 + 0xfffffc - 0x27 = 0xffffec

which is

.. literalinclude:: ./code/day2/ex1/hello.S
  :caption: relocation table of hello.o
  :lineno-start: 185
  :lines: 185-186
  :linenos:


.. WARNING::

  ``xxx`` is a relative offset and is NOT an absolute offset.


.. literalinclude:: ./code/day2/ex1/hello.S
  :caption: relocation table of hello.o
  :lineno-start: 110
  :lines: 110-121
  :linenos:

.. literalinclude:: ./code/day2/ex1/hello.S
  :caption: relocation table of hello.o
  :lineno-start: 188
  :lines: 188-190
  :linenos:

Default linker script
^^^^^^^^^^^^^^^^^^^^^

``--verbose`` will show the default linker script.

.. code-block::

  ld -m elf_i386 -o hello --verbose hello.o > default-linker-script.ld

.. literalinclude:: ./code/day2/ex1/default-linker-script.ld
  :caption: default linker script
  :linenos:
