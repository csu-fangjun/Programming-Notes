
ELF Note
========

.. literalinclude:: ./code/elf_note/hello.s
  :caption: hello.s
  :linenos:

.. literalinclude:: ./code/elf_note/Makefile
  :caption: Makefile
  :linenos:

.. code-block::

  readelf -S hello.o

prints

.. code-block::

    There are 9 section headers, starting at offset 0x80:

    Section Headers:
      [Nr] Name              Type            Addr     Off    Size   ES Flg Lk Inf Al
      [ 0]                   NULL            00000000 000000 000000 00      0   0  0
      [ 1] .text             PROGBITS        00000000 000034 00000c 00  AX  0   0  1
      [ 2] .rel.text         REL             00000000 000260 000008 08      7   1  4
      [ 3] .data             PROGBITS        00000000 000040 000000 00  WA  0   0  1
      [ 4] .bss              NOBITS          00000000 000040 000000 00  WA  0   0  1
      [ 5] .text.foo         PROGBITS        00000000 000040 000006 00  AX  0   0  1
      [ 6] .shstrtab         STRTAB          00000000 000046 00003a 00      0   0  1
      [ 7] .symtab           SYMTAB          00000000 0001e8 000070 10      8   5  4
      [ 8] .strtab           STRTAB          00000000 000258 000008 00      0   0  1
    Key to Flags:
      W (write), A (alloc), X (execute), M (merge), S (strings)
      I (info), L (link order), G (group), T (TLS), E (exclude), x (unknown)
      O (extra OS processing required) o (OS specific), p (processor specific)


Note that ``.text.foo`` will be merged into ``.text`` by the linker script:

.. literalinclude:: ./code/day2/ex1/default-linker-script.ld
  :caption: part of the default linker script
  :lineno-start: 72
  :lines: 72-81
  :linenos:

.. literalinclude:: ./code/day2/ex1/default-linker-script.ld
  :caption: default linker script
  :linenos:

init_array
----------

.. literalinclude:: ./code/elf_note/init.cc
  :caption: init.cc
  :linenos:

.. NOTE::

  ``Foo f`` is a global object with constructor and destructor

It defines ``.section .init_array,"aw"``, which contains a pointer to
the function ``_GLOBAL__sub_I_f``. Inside ``_GLOBAL__sub_I_f``, it first calls
the constructor of ``Foo()`` for ``f``, and then calls ``__cxa_atexit(&Foo::~Foo(), &f, &__dso_handle, )``.

``__cxa_atexit`` is defined in `<https://itanium-cxx-abi.github.io/cxx-abi/abi.html#dso-dtor-motivation>`_.

.. code-block::

  extern "C" int __cxa_atexit ( void (*f)(void *), void *p, void *d );


.. literalinclude:: ./code/elf_note/init.s
  :caption: init.s
  :linenos:

Note that
- ``__attribute__((constructor))`` is compiled to ``.section	.init_array``
- ``__attribute__((destructor))`` is compiled to ``.section	.fini_array``

.. literalinclude:: ./code/elf_note/constructor.cc
  :caption: constructor.cc
  :linenos:

.. literalinclude:: ./code/elf_note/constructor.s
  :caption: constructor.s
  :linenos:
