
Bison
=====

Bison is an open source project and can be used as a substitute of yacc.

To install it, use

.. code-block::

  sudo apt-get install bison

Although bison replaces ``yacc``, we still need to link to ``yacc``.
It is ``liby.a``, which is in ``/usr/lib/x86_64-linux-gnu``.

.. code-block::

    %{
    %}

    %start
    %token
    %union
    %type

.. code-block::

  statements: statement {printf("statement");}
    | statement statements {printf("statements\"n);}

  statement:  identifier '+' identifier  {printf("plus\n");}
  statement:  identifier '-' identifier  {printf("minus\n");}

