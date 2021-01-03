Macros
======

Identify GCC and Clang
----------------------

See `<https://github.com/google/sanitizers/wiki/AddressSanitizer>`_

.. code-block::

  #if defined(__clang__) || defined (__GNUC__)
  # define ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
  #else
  # define ATTRIBUTE_NO_SANITIZE_ADDRESS
  #endif

See `<https://github.com/facebook/folly/blob/master/folly/Portability.h#L25>`_

.. code-block::

  #if defined(__GNUC__) && !defined(__clang__)
  static_assert(__GNUC__ >= 5, "__GNUC__ >= 5");
  #endif


See `<https://github.com/facebook/folly/blob/master/folly/CPortability.h#L28>`_

.. code-block::

  #if defined __GNUC__ && defined __GNUC_MINOR__
  #define __GNUC_PREREQ(maj, min) \
    ((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
  #else
  #define __GNUC_PREREQ(maj, min) 0
  #endif

See `<https://github.com/facebook/folly/blob/master/folly/CPortability.h#L40>`_

.. code-block::

  #if defined __clang__ && defined __clang_major__ && defined __clang_minor__
  #define __CLANG_PREREQ(maj, min) \
    ((__clang_major__ << 16) + __clang_minor__ >= ((maj) << 16) + (min))
  #else
  #define __CLANG_PREREQ(maj, min) 0
  #endif

has_builtin
-----------

See `<https://github.com/facebook/folly/blob/master/folly/CPortability.h#L50>`_

.. code-block::

  #if defined(__has_builtin)
  #define FOLLY_HAS_BUILTIN(...) __has_builtin(__VA_ARGS__)
  #else
  #define FOLLY_HAS_BUILTIN(...) 0
  #endif

has_feature
-----------

See `<https://github.com/facebook/folly/blob/master/folly/CPortability.h#L56>`_

.. code-block::

  #if defined(__has_feature)
  #define FOLLY_HAS_FEATURE(...) __has_feature(__VA_ARGS__)
  #else
  #define FOLLY_HAS_FEATURE(...) 0
  #endif

export
------

See `<https://github.com/facebook/folly/blob/master/folly/CPortability.h#L161>`_

.. code-block::

  #if defined(__GNUC__)
  #define FOLLY_EXPORT __attribute__((__visibility__("default")))
  #else
  #define FOLLY_EXPORT
  #endif

hidden
------

See `<https://github.com/facebook/folly/blob/master/folly/CPortability.h#L186>`_

.. code-block::

  #if defined(_MSC_VER)
  #define FOLLY_ATTR_VISIBILITY_HIDDEN
  #elif defined(__GNUC__)
  #define FOLLY_ATTR_VISIBILITY_HIDDEN __attribute__((__visibility__("hidden")))
  #else
  #define FOLLY_ATTR_VISIBILITY_HIDDEN
  #endif

noinline
--------

See `<https://github.com/facebook/folly/blob/master/folly/CPortability.h#L168>`_

.. code-block::

  #ifdef _MSC_VER
  #define FOLLY_NOINLINE __declspec(noinline)
  #elif defined(__GNUC__)
  #define FOLLY_NOINLINE __attribute__((__noinline__))
  #else
  #define FOLLY_NOINLINE
  #endif

always inline
-------------

See `<https://github.com/facebook/folly/blob/master/folly/CPortability.h#L177>`_

.. code-block::

  #ifdef _MSC_VER
  #define FOLLY_ALWAYS_INLINE __forceinline
  #elif defined(__GNUC__)
  #define FOLLY_ALWAYS_INLINE inline __attribute__((__always_inline__))
  #else
  #define FOLLY_ALWAYS_INLINE inline
  #endif

weak
----

See `<https://github.com/facebook/folly/blob/master/folly/CPortability.h#L195>`_

.. code-block::

  #if FOLLY_HAVE_WEAK_SYMBOLS
  #define FOLLY_ATTR_WEAK __attribute__((__weak__))
  #else
  #define FOLLY_ATTR_WEAK
  #endif

little endian
-------------

See `<https://github.com/facebook/folly/blob/master/folly/Portability.h#L332>`_

.. code-block::

  constexpr auto kIsLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
