
Flex
====

flex is an open source implementation of ``lex``.

To install it, use

.. code-block::

  sudo apt-get install flex


Hello world
-----------

.. literalinclude:: ./code/flex/hello/hello.l
  :caption: hello.l
  :language: c
  :linenos:

.. literalinclude:: ./code/flex/hello/Makefile
  :caption: Makefile
  :language: makefile
  :linenos:

Note that ``/usr/bin/lex`` is a symlink to ``/usr/bin/flex``

.. code-block::

  lrwxrwxrwx 1 root root 4 Feb 28  2016 /usr/bin/lex -> flex

``libl.a`` is a symlink to ``libfl.a``:

.. code-block::

  lrwxrwxrwx 1 root root 7 Feb 28  2016 /usr/lib/x86_64-linux-gnu/libl.a -> libfl.a

.. note::

  1. We can use either ``lex`` or ``flex`` to compile ``*.l`` files.
  2. We can link to either ``-ll`` or ``-lfl``.
  3. The default output filename of ``lex`` is ``lex.yy.c``.

.. code-block::

  $ nm /usr/lib/x86_64-linux-gnu/libfl.a

  libmain.o:
  0000000000000000 T main
                   U yylex

  libyywrap.o:
  0000000000000000 T yywrap

The disassemble output of `libyywrap.o` is:

.. code-block::

    libyywrap.o:     file format elf64-x86-64

    Disassembly of section .text:

    0000000000000000 <yywrap>:
       0:	b8 01 00 00 00       	mov    $0x1,%eax
       5:	c3                   	retq

We can see that ``yywrap`` just returns 1.

The disassemble of ``libmain.o`` is:

.. code-block::

    libmain.o:     file format elf64-x86-64

    Disassembly of section .text.startup:

    0000000000000000 <main>:
       0:	48 83 ec 08          	sub    $0x8,%rsp
       4:	0f 1f 40 00          	nopl   0x0(%rax)
       8:	e8 00 00 00 00       	callq  d <main+0xd>
       d:	85 c0                	test   %eax,%eax
       f:	75 f7                	jne    8 <main+0x8>
      11:	48 83 c4 08          	add    $0x8,%rsp
      15:	c3                   	retq

Its symbol table is ``nm libmain.o``::

    0000000000000000 T main
                     U yylex

And its relocation table is ``readelf -r libmain.o``::

    Relocation section '.rela.text.startup' at offset 0x188 contains 1 entries:
      Offset          Info           Type           Sym. Value    Sym. Name + Addend
    000000000009  000900000002 R_X86_64_PC32     0000000000000000 yylex - 4

    Relocation section '.rela.eh_frame' at offset 0x1a0 contains 1 entries:
      Offset          Info           Type           Sym. Value    Sym. Name + Addend
    000000000020  000100000002 R_X86_64_PC32     0000000000000000 .text.startup + 0

which is equivalent to::

    extern int yylex();
    int main() {
      while(yylex()) {}
      return 0;
    }

To run ``./hello``, we can use:

  1. ``./hello``, then enter any string ending with a newline, it will print something.
  2. ``printf "hello world\nexit" | ./hello``, it will print

  .. code-block::

    Good bye.
    exiting

.. NOTE::

  Anything that is not matched by the pattern is copied to output by default.

count lines
-----------

.. literalinclude:: ./code/flex/count_lines/count.l
  :caption: count.l
  :language: c
  :linenos:

.. literalinclude:: ./code/flex/count_lines/count2.l
  :caption: count2.l
  :language: c
  :linenos:

.. literalinclude:: ./code/flex/count_lines/count3.l
  :caption: count3.l
  :language: c
  :linenos:

.. note::

  We use ``yytext`` to print the matched string; ``yylen`` is the length of ``yytext``,
  which is ``strlen(yytext)``.

.. literalinclude:: ./code/flex/detect/detect.l
  :caption: detect.l
  :language: flex
  :linenos:

Basics
------


- ``[0-9]`` is equivalent to ``0 | 1 | 2 | 3 | 4 | 5 | 7 | 8 | 9``
- ``[^0-9]`` matches any non-digit character
- The dot ``.`` matches any character **except** the newline
- ``x*`` zero or more occurrences of ``x``
- ``x?`` zero or one occurrence of ``x``
- ``x+`` one ore more occurrences of ``x``
- ``x{n, m}``  between ``n`` and ``m``
- ``^x`` matches x at the beginning of a line
- ``x$`` matches x at the end of a line
- ``{name}`` reuses the previously defined patter ``name``.
- ``.*`` zero or more characters
- ``.+`` one or more characters
- ``rs|tu`` ``rs`` or ``tu``
- ``a(b|c)d``  ``abd`` or ``acd``
- character classes::

    [:alnum:]  [:alpha:]  [:blank:]
    [:cntrl:]  [:digit:]  [:graph:]
    [:lower:]  [:print:]  [:punct:]
    [:space:]  [:upper:]  [:xdigit:]

- ``[[:alnum:]]`` is equivalent to::

    [[::alpha:][:digit]]
    [[:alpha:][0-9]]
    [a-zA-Z0-9]

- There exists also::

    [:^alnum:]  [:^alpha:]   .......

.. warning::

    [:alnum:] is wrapped in [], that is, we use [[:alnum:]]

- ``[a-c]{-}[b-z]``, the set different of ``[a-c]]`` minus ``[b-z]``. ``{-}`` means set differences between two character classes
- ``[a-c]{+}[b-z]``, set union between ``[a-c]`` and ``[b-z]``.

It will match against the input string as long as possible. The earliest
patter has a high priority.


The function ``yylex()`` parses one token and returns an ``int`` and it will set several
global variables:

- ``yytext``, it is of type ``char*``, pointing to the token. It is overwritten every time ``yylex()`` is called.
- ``yyleng``, it is of type ``int``, which equals to ``strlen(yytext)``.

When using flex together with ``bison``, ``yylex()`` returns an integer defined by ``bison``.
This integer starts from 258. Integer zero means end of file.

``ECHO`` is a C macro defined as

.. code-block::

  #define ECHO do { if (fwrite( yytext, yyleng, 1, yyout )) {} } while (0)

References
----------

- `<http://dinosaur.compilertools.net/flex/manpage.html>`_

    Manual for flex.

- ``info --vi-keys flex``

