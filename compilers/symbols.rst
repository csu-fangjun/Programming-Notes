
Symbols
=======

Weak Symbols
------------

.. code-block::

  __attribute__((weak)) int hello();
  __attribute__((weak)) int hello2() {}

  void test() {
    hello();
    hello2();
  }

  __attribute__((weak)) int foo;


``nm -C test.o`` prints:

.. code-block::

  w hello()
  W hello2()
  V foo

Note that it is lowercase ``w`` for ``hello`` since it does not have an implementation;
while it is uppercase ``W`` for ``hello2``.

`V` means object.

bss
---

.. code-block::

  int a;
  static int b;
  void test() {}

- ``B a``
- ``b b``

data
----

.. code-block::

  int a = 10;
  static int b = 10;
  void test() {}

- ``D a``
- ``d b``

rodata
------

.. code-block::

  const int a = 10;
  static const int b = 10;
  extern const int c = 1;
  void test() {}

- ``r a``
- ``r b``
- ``R c``

Note that ``const`` objects are implicitly ``local``.

text
----

.. code-block::

  void test1() {}
  static void test2() {}

- ``T test1()``
- ``t test2()``

alias
-----

.. code-block::

  extern "C" {
    void test() {}
    void test2() __attribute__((weak, alias("test")));
  }

- ``T test``
- ``W test2``




References
----------

- Understand Weak Symbols by Examples

    `<http://winfred-lu.blogspot.com/2009/11/understand-weak-symbols-by-examples.html>`_
