
type_traits
===========

``#include<type_traits>``

See `<https://en.cppreference.com/w/cpp/header/type_traits>`_

integral_constant
-----------------

See `<https://en.cppreference.com/w/cpp/types/integral_constant>`_

Its also includes:

  - ``bool_constant``
  - ``true_type``
  - ``false_type``

.. literalinclude:: ./code/type_traits/integral_constant.h
  :caption: code/type_traits/integral_constant.h
  :language: cpp
  :linenos:

is_same
-------
.. literalinclude:: ./code/type_traits/is_same.h
  :caption: code/type_traits/is_same.h
  :language: cpp
  :linenos:

remove_const
------------

See `<https://en.cppreference.com/w/cpp/types/remove_cv>`_

.. literalinclude:: ./code/type_traits/remove_const.h
  :caption: code/type_traits/remove_const.h
  :language: cpp
  :linenos:

remove_volatile
---------------

See `<https://en.cppreference.com/w/cpp/types/remove_cv>`_

.. literalinclude:: ./code/type_traits/remove_volatile.h
  :caption: code/type_traits/remove_volatile.h
  :language: cpp
  :linenos:

remove_cv
---------

See `<https://en.cppreference.com/w/cpp/types/remove_cv>`_

.. literalinclude:: ./code/type_traits/remove_cv.h
  :caption: code/type_traits/remove_cv.h
  :language: cpp
  :linenos:

remove_reference
----------------

See `<https://en.cppreference.com/w/cpp/types/remove_reference>`_

.. literalinclude:: ./code/type_traits/remove_reference.h
  :caption: code/type_traits/remove_reference.h
  :language: cpp
  :linenos:

is_reference
------------

See `<https://en.cppreference.com/w/cpp/types/is_reference>`_

.. literalinclude:: ./code/type_traits/is_reference.h
  :caption: code/type_traits/is_reference.h
  :language: cpp
  :linenos:

is_lvalue_reference
-------------------

See `<https://en.cppreference.com/w/cpp/types/is_lvalue_reference>`_

.. literalinclude:: ./code/type_traits/is_lvalue_reference.h
  :caption: code/type_traits/is_lvalue_reference.h
  :language: cpp
  :linenos:

is_rvalue_reference
-------------------

See `<https://en.cppreference.com/w/cpp/types/is_rvalue_reference>`_

.. literalinclude:: ./code/type_traits/is_rvalue_reference.h
  :caption: code/type_traits/is_rvalue_reference.h
  :language: cpp
  :linenos:

is_void
-------

See `<https://en.cppreference.com/w/cpp/types/is_void>`_

.. literalinclude:: ./code/type_traits/is_void.h
  :caption: code/type_traits/is_void.h
  :language: cpp
  :linenos:

is_pointer
----------

Note that it can be:

  - a pointer to an array
  - a pointer to a function
  - a pointer to a variable

But not a pointer to a data member or a member function.

See `<https://en.cppreference.com/w/cpp/types/is_pointer>`_

.. literalinclude:: ./code/type_traits/is_pointer.h
  :caption: code/type_traits/is_pointer.h
  :language: cpp
  :linenos:


is_member_pointer
-----------------

See `<https://en.cppreference.com/w/cpp/types/is_member_pointer>`_

.. literalinclude:: ./code/type_traits/is_member_pointer.h
  :caption: code/type_traits/is_member_pointer.h
  :language: cpp
  :linenos:

is_const
--------

See `<https://en.cppreference.com/w/cpp/types/is_const>`_

.. literalinclude:: ./code/type_traits/is_const.h
  :caption: code/type_traits/is_const.h
  :language: cpp
  :linenos:

is_array
--------

See `<https://en.cppreference.com/w/cpp/types/is_array>`_

.. literalinclude:: ./code/type_traits/is_array.h
  :caption: code/type_traits/is_array.h
  :language: cpp
  :linenos:

rank
----

See

  - `<https://en.cppreference.com/w/cpp/types/rank>`_

.. literalinclude:: ./code/type_traits/rank.h
  :caption: code/type_traits/rank.h
  :language: cpp
  :linenos:


.. literalinclude:: ./code/type_traits/extent_test.cc
  :caption: code/type_traits/extent_test.cc
  :language: cpp
  :linenos:

extent
------

See

  - `<https://en.cppreference.com/w/cpp/types/extent>`_

.. literalinclude:: ./code/type_traits/extent.h
  :caption: code/type_traits/extent.h
  :language: cpp
  :linenos:


remove_extent
-------------

See

  - `<https://en.cppreference.com/w/cpp/types/remove_extent>`_

.. literalinclude:: ./code/type_traits/remove_extent.h
  :caption: code/type_traits/remove_extent.h
  :language: cpp
  :linenos:

constructible
-------------

The core part is ``std::is_constructible``.

See

  - ``std::is_trivial``: `<https://en.cppreference.com/w/cpp/types/is_trivial>`_
  - `<https://en.cppreference.com/w/cpp/named_req/TrivialType>`_
  - ``std::is_trivially_copyable``: `<https://en.cppreference.com/w/cpp/types/is_trivially_copyable>`_
  - `<https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable>`_
  - ``std::is_constructible``: `<https://en.cppreference.com/w/cpp/types/is_constructible>`_

    .. NOTE::

      For ``std::is_trivially_constructible_v`` to be true, ``T`` can be
      a scalar type or T has a default copy constructor (summarized by fangjun).

   - ``std::is_default_constructible``, ``std::is_trivially_default_constructible``,
     ``std::is_trivially_constructible``.

   - ``std::is_copy_constructible``, ``std::is_trivially_copy_constructible``

   - ``std::is_move_constructible``, ``std::is_trivially_move_constructible``

.. literalinclude:: ./code/type_traits/is_member_pointer.h
  :caption: code/type_traits/is_member_pointer.h
  :language: cpp
  :linenos:

void_t
------

See `<https://en.cppreference.com/w/cpp/types/void_t>`_

.. literalinclude:: ./code/type_traits/void_t.h
  :caption: code/type_traits/void_t.h
  :language: cpp
  :linenos:

is_assignable
-------------

See

  - ``std::is_assignable``, ``std::is_trivially_assignable``,
    `<https://en.cppreference.com/w/cpp/types/is_assignable>`_

  - ``std::is_copy_assignable``, ``std::is_trivially_copy_assignable``,
    `<https://en.cppreference.com/w/cpp/types/is_copy_assignable>`_

  - ``std::is_move_assignable``, ``std::is_trivially_move_assignable``,
    `<https://en.cppreference.com/w/cpp/types/is_move_assignable>`_

      The can be implemented with the help of ``std::is_assignable``.

.. literalinclude:: ./code/type_traits/is_assignable.h
  :caption: code/type_traits/is_assignable.h
  :language: cpp
  :linenos:

is_nullptr
----------

See `<https://en.cppreference.com/w/cpp/types/is_nullptr>`_

.. literalinclude:: ./code/type_traits/is_nullptr.h
  :caption: code/type_traits/is_nullptr.h
  :language: cpp
  :linenos:
