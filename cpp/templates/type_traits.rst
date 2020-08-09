
Type Traits
===========

Header Files
------------
- ``/usr/include/c++/5.4.0/type_traits``
- `<https://github.com/llvm-mirror/libcxx/blob/master/include/type_traits>`_
- `<https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/std/type_traits>`_
- `<https://github.com/microsoft/STL/blob/master/stl/inc/type_traits>`_

- type classification traits
- type property inspection traits
- type transformation traits

Papers
------

- `A Proposal to add Type Traits to the Standard Library (2003) <http://open-std.org/jtc1/sc22/wg21/docs/papers/2003/n1424.htm>`_



add_rvalue_reference
--------------------

.. literalinclude:: ./code/type_traits/add_rvalue_reference.cc
   :caption: add_rvalue_reference.cc
   :language: cpp
   :linenos:

.. WARNING::

  The return type of ``add_rvalue_reference`` is ``T&&``,
  which can be either a left value reference or a right value
  reference. The name ``add_rvalue_reference`` is misleading!
