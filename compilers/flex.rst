
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

To run ``./hello``, we can use:

  1. ``./hello``, then enter any string ending with a newline, it will print something.
  2. ``printf "hello worldexit"``, it will print

  .. code-block::

    Good bye.
    exiting

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

