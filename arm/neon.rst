
.. toctree::
  :maxdepth: 5

NEON
====

NEON is just a brand name from ARM. Its essence is SIMD.


Intrinsics
----------

arm_neon.h
::::::::::

Before using neon intrinsics, we have to include the header ``arm_neon.h``.
However, before including the header, we have to check whether
the compiler supports it.

According to `<http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0491c/BABJFEFG.html>`_,
we have to use the macro ``__ARM_NEON__``:

  This macro can be used to conditionally include ``arm_neon.h``, to permit the use of NEON intrinsics.

For the cross toolchain ``aarch64-linux-gnu-g++``:

.. code-block:: console

  $ aarch64-linux-gnu-g++ -E -dM - < /dev/null | grep -i neon

prints::

  #define __ARM_NEON_FP 12
  #define __ARM_NEON 1

For a native gcc for arm:

.. code-block:: console

  $ file /usr/bin/gcc-4.8
  /usr/bin/gcc-4.8: ELF 32-bit LSB  executable, ARM, EABI5 version 1 (SYSV),
  dynamically linked (uses shared libs), for GNU/Linux 2.6.32,
  BuildID[sha1]=04147ebb341f6752cff10a1d1bf89a44cafb2ff9, stripped

The output of the command:

.. code-block:: console

  $ gcc -E -dM - < /dev/null | grep --color -i neon

is::

  #define __ARM_NEON_FP 4

The output of the following command with ``-mfpu=neon``:

.. code-block:: console

  $ gcc -E -dM -mfpu=neon - < /dev/null | grep --color -i neon

is::

  #define __ARM_NEON_FP 4
  #define __ARM_NEON__ 1
  #define __ARM_NEON 1

Therefore, we can use either ``__ARM_NEON`` or ``__ARM_NEON__``:

.. code-block:: cpp

  #if defined(__ARM_NEON__) || defined(__ARM_NEON)
    #include <arm_neon.h>
  #endif

If we really need to include ``arm_neon.h``, we can define these two macros manually
while invoking gcc::

  gcc -D__ARM_NEON__ -D__ARM_NEON


.. HINT::

  This page `<https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/neon-programmers-guide-for-armv8-a/optimizing-c-code-with-neon-intrinsics/single-page>`_ says:

    **__ARM_NEON**
        - Advanced SIMD is supported by the compiler
        - Always 1 for AArch64


``gcc`` has two versions of ``arm_neon.h``:

  - For 32-bit arm: `<https://github.com/gcc-mirror/gcc/blob/master/gcc/config/arm/arm_neon.h>`_

  - For 64-bit arm: `<https://github.com/gcc-mirror/gcc/blob/master/gcc/config/aarch64/arm_neon.h>`_

      This file is over 1MB and GitHub cannot render it. Refer to
      `<https://raw.githubusercontent.com/gcc-mirror/gcc/master/gcc/config/aarch64/arm_neon.h>`_
      for a raw text version.

  - There is another version for 64-bit arm:

      `<https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h>`_

      It provides inline assembly implementations for the intrinsics!

``clang`` handles 32-bit arm and 64-bit arm in a single ``arm_neon.h``:
`<https://github.com/Cronus-Emulator/clang/blob/master/include/arm_neon.h>`

    It checks inside the file:
    `<https://github.com/Cronus-Emulator/clang/blob/d37ba7b2d93b5d582beb238d9820c521534c7807/include/arm_neon.h#L27>`_

      .. code-block:: cpp

        #if !defined(__ARM_NEON__) && !defined(__ARM_NEON)
        #error "NEON support not enabled"
        #endif

Basic Data Types
----------------


In `<https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#32>`_,
it defines ``int8x8_t`` as:

.. code-block:: cpp

  typedef __builtin_aarch64_simd_qi int8x8_t
    __attribute__ ((__vector_size__ (8)));

According to `<https://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html>`_:

  Vectors can be subscripted as if the vector were an array with the same number of elements and base type.
  Out of bound accesses invoke undefined behavior at run time.


Clang defines ``int8x8_t`` as `<https://github.com/Cronus-Emulator/clang/blob/d37ba7b2d93b5d582beb238d9820c521534c7807/include/arm_neon.h#L47>`_:

.. code-block:: cpp

  typedef __attribute__((neon_vector_type(8)))  int8_t int8x8_t;

And according to `<https://clang.llvm.org/docs/LanguageExtensions.html#vector-operations>`_, we cannot use ``[]`` to index
a single element in ``int8x8_t``.

+-------------+--------------------------------+
| Register    | C Types                        |
+=============+================================+
| v0.8b       | ``int8x8_t``, ``uint8x8_t``    |
+-------------+--------------------------------+
| v0.16b      | ``int8x16_t``, ``uint8x16_t``  |
+-------------+--------------------------------+
| v0.4h       | ``int16x4_t``, ``uint16x4_t``  |
+-------------+--------------------------------+
| v0.8h       | ``int16x8_t``, ``uint16x8_t``  |
+-------------+--------------------------------+
| v0.2s       | ``int32x2_t``, ``uint32x2_t``  |
+-------------+--------------------------------+
| v0.4s       | ``int32x4_t``, ``uint32x4_t``  |
+-------------+--------------------------------+
| v0.1d       | ``int64x1_t``, ``uint64x1_t``  |
+-------------+--------------------------------+
| v0.2d       | ``int64x2_t``, ``uint64x2_t``  |
+-------------+--------------------------------+

Intrinsics Implementations
--------------------------

``vld1q_s8``
::::::::::::

Refer to `[1]`_

.. _[1]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#9142

.. code-block:: cpp

  __extension__ static __inline int8x16_t __attribute__ ((__always_inline__))
    vld1q_s8 (const int8_t * a)
  {
      int8x16_t result;
      __asm__ ("ld1 {%0.16b}, %1"
                : "=w"(result)
                : "Utv"(({const int8x16_t *_a = (int8x16_t *) a; *_a;}))
                : /* No clobbers */);
      return result;
  }

According to `[2]`_:

  **Utv**:
    An address valid for loading/storing opaque structure
    types wider than TImode.

.. _[2]: https://github.com/gcc-mirror/gcc/blob/705510a708d3642c9c962beb663c476167e4e8a4/gcc/config/aarch64/constraints.md#L310

``vmulq_n_s32``
:::::::::::::::

Refer to `[3]`_.

.. _[3]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#12743

.. code-block:: cpp

  __extension__ static __inline int32x4_t __attribute__ ((__always_inline__))
  vmulq_n_s32 (int32x4_t a, int32_t b)
  {
    int32x4_t result;
    __asm__ ("mul %0.4s,%1.4s,%2.s[0]"
             : "=w"(result)
             : "w"(a), "w"(b)
             : /* No clobbers */);
    return result;
  }

The meaning of the constraint ``w`` is: `[4]`_

    Floating point and SIMD vector registers.

.. _[4]: https://github.com/gcc-mirror/gcc/blob/705510a708d3642c9c962beb663c476167e4e8a4/gcc/config/aarch64/constraints.md#L27


``vmulq_s32``
:::::::::::::

Refer to `[5]`_

.. code-block:: cpp

  __extension__ static __inline int32x4_t __attribute__ ((__always_inline__))
  vmulq_s32 (int32x4_t __a, int32x4_t __b)
  {
    return __a * __b;
  }

.. _[5]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#1118

``vmlaq_s32``
:::::::::::::

``ret = a + b*c``.

Refer to `[6]`_:


.. code-block:: cpp

  __extension__ static __inline int32x4_t __attribute__ ((__always_inline__))
  vmlaq_s32 (int32x4_t a, int32x4_t b, int32x4_t c)
  {
    int32x4_t result;
    __asm__ ("mla %0.4s, %2.4s, %3.4s"
             : "=w"(result)
             : "0"(a), "w"(b), "w"(c)
             : /* No clobbers */);
    return result;
  }

.. _[6]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#10437

``vmlaq_n_s32``
:::::::::::::::

Refer to `[7]`_:

.. code-block:: cpp

  __extension__ static __inline int32x4_t __attribute__ ((__always_inline__))
  vmlaq_n_s32 (int32x4_t a, int32x4_t b, int32_t c)
  {
    int32x4_t result;
    __asm__ ("mla %0.4s,%2.4s,%3.s[0]"
             : "=w"(result)
             : "0"(a), "w"(b), "w"(c)
             : /* No clobbers */);
    return result;
  }

.. _[7]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#10382

``vrecpeq_f32``
:::::::::::::::

Take the reciprocal of a vector.

Refer to `[8]`_:

.. code-block:: cpp

  __extension__ static __inline float32x4_t __attribute__ ((__always_inline__))
  vrecpeq_f32 (float32x4_t a)
  {
    float32x4_t result;
    __asm__ ("frecpe %0.4s,%1.4s"
             : "=w"(result)
             : "w"(a)
             : /* No clobbers */);
    return result;
  }

.. _[8]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#14593

It turns out ``vrecpeq_f32`` is not very accurate. A much more accurate method is
the Newton method. Suppose we have a point :math:`(x_0, f(x_0))` and the derivative
at :math:`x_0` is :math:`f'(x_0)`. To find :math:`(x, 0)`, we have

.. math::

  \frac{f(x_0) - 0}{x_0 - x} = f'(x_0)

which can be simplified to

.. math::

  x = x_0 - \frac{f(x_0)}{f'(x_0)}


To find the reciprocal of ``d``, let

.. math::

  f(x) = \frac{1}{x} - d

It's easy to see that :math:`f(\frac{1}{1/d}) == 0`
and :math:`f'(x) = -\frac{1}{x^2}`.


Therefore, we get

.. math::

  x = x_0 - \frac{f(x_0)}{f'(x_0)} = x_0 - \frac{1/x_0 - d}{-1/x_0^2} = x_0+x_0^2(\frac{1}{x_0}-d) = 2x_0 -dx_0^2 = (2-dx_0)x_0

There is an intrinsic function ``vrecpsq_f32(a, b)`` that computes ``2 - a*b``

The following is an example of using ``vrecpeq_f32`` and ``vrecpsq_f32`` to perform
two iterations of the Newton method.

.. code-block:: cpp

  void test_reciprocal() {
    float32x4_t a = {1, 2, 3, 5};
    float32x4_t b = vrecpeq_f32(a); // b = 1/a
    // printf("%.6f, %.6f, %.6f, %.6f\n", b[0], b[1], b[2], b[3]);
    // 0.998047, 0.499023, 0.333008, 0.199707

    // perform one iteration
    float32x4_t c = vrecpsq_f32(a, b); // c = 2 - a*b
    b = vmulq_f32(c, b);
    // printf("%.6f, %.6f, %.6f, %.6f\n", b[0], b[1], b[2], b[3]);
    // 0.999996, 0.499998, 0.333333, 0.200000

    // perform another iteration
    c = vrecpsq_f32(a, b);
    b = vmulq_f32(c, b);
    // printf("%.6f, %.6f, %.6f, %.6f\n", b[0], b[1], b[2], b[3]);
    // 1.000000, 0.500000, 0.333333, 0.200000

    // 2-1*1=1, 2-2*2=-2, 2-3*3=-7, 2-5*5=-23
    c = vrecpsq_f32(a, a);
    // printf("%.6f, %.6f, %.6f, %.6f\n", c[0], c[1], c[2], c[3]);
    // 1.000000, -2.000000, -7.000000, -23.000000
  }

More information about Newton's method can be found
at
`CS3343/3341 Analysis of Algorithms  Newton's Method to Perform Division <http://www.cs.utsa.edu/~wagner/CS3343/newton/division.html>`_.

``vld1q_dup_s32``
:::::::::::::::::

Refer to `[9]`_.

.. code-block:: cpp

  __extension__ static __inline int32x4_t __attribute__ ((__always_inline__))
  vld1q_dup_s32 (const int32_t * a)
  {
    int32x4_t result;
    __asm__ ("ld1r {%0.4s}, %1"
             : "=w"(result)
             : "Utv"(*a)
             : /* No clobbers */);
    return result;
  }

``ld1r``:
  Load single 1-element structure and replicate to all lanes (of one register).

  http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0491c/BABJFEFG.html

.. _[9]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#8876

``vmovq_n_s32``
:::::::::::::::

Refer to `[10]`_.

.. code-block:: cpp

  __extension__ static __inline int32x4_t __attribute__ ((__always_inline__))
  vmovq_n_s32 (int32_t a)
  {
    int32x4_t result;
    __asm__ ("dup %0.4s,%w1"
             : "=w"(result)
             : "r"(a)
             : /* No clobbers */);
    return result;
  }

.. _[10]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#11840

``vst1q_s32``
:::::::::::::

Refer to `[11]`_.

.. code-block:: cpp

  __extension__ static __inline void __attribute__ ((__always_inline__))
  vst1q_s32 (int32_t * a, int32x4_t b)
  {
    __asm__ ("st1 {%1.4s},[%0]"
             :
             : "r"(a), "w"(b)
             : "memory");
  }


.. _[11]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#16715


``vmlaq_f32``
:::::::::::::

Refer to `[12]`_.

.. code-block:: cpp

  __extension__ static __inline float32x4_t __attribute__ ((__always_inline__))
  vmlaq_f32 (float32x4_t a, float32x4_t b, float32x4_t c)
  {
    return a + b * c;
  }

.. _[12]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#21611

The implementation of ``vmlaq_f32`` is not efficient. A more efficient instruction
is ``vfmaq_f32`` `[13]`_:

.. code-block:: cpp

  __extension__ static __inline float32x4_t __attribute__ ((__always_inline__))
  vfmaq_f32 (float32x4_t a, float32x4_t b, float32x4_t c)
  {
    float32x4_t result;
    __asm__ ("fmla %0.4s,%2.4s,%3.4s"
             : "=w"(result)
             : "0"(a), "w"(b), "w"(c)
             : /* No clobbers */);
    return result;
  }

.. _[13]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#7827

NNPack defines its own ``vmuladdq_f32`` `[14]`_:

.. code-block:: cpp

  static inline float32x4_t vmuladdq_f32(float32x4_t acc, float32x4_t a, float32x4_t b) {
    #if defined(__aarch64__)
      return vfmaq_f32(acc, a, b);
    #else
      return vmlaq_f32(acc, a, b);
    #endif
  }

.. _[14]: https://github.com/Maratyszcza/NNPACK/blob/bda381b2e207230cab2b38be310a17831cfe384a/include/nnpack/arm_neon.h#L65

Similar versions exist for:

  - ``vmlsq_f32`` `[15]`_ and ``vfmsq_f32`` `[16]`_ (multiply and subtract).
  - ``vmlaq_lane_f32`` `[17]`_ and ``vfmaq_lane_f32`` `[18]`_
  - ``vmlsq_lane_f32`` `[19]`_ and ``vfmsq_lane_f32`` `[20]`_

.. _[15]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#21629
.. _[16]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#7947
.. _[17]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#10219
.. _[18]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#7848
.. _[19]: https://android.googlesource.com/toolchain/gcc/+/32fce3edda831e36ee484406c39dffbe0230f257/gcc-4.8/gcc/config/aarch64/arm_neon.h#11117
.. _[20]: https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics?search=vfmsq_lane_f32






References
----------

- Optimizing C Code with Neon Intrinsics - single page
  `<https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/neon-programmers-guide-for-armv8-a/optimizing-c-code-with-neon-intrinsics/single-page>`_

    It gives an overview about NEON intrinsics.


- `<https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics>`_

      It provides a search box to search help info for intrinsic functions.

- Introducing Neon for Armv8-A - single page
  `<https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/neon-programmers-guide-for-armv8-a/introducing-neon-for-armv8-a/single-page>`_

    It gives a very good overview of NEON.

- NEON™ Version: 1.0 Programmer’s Guide
  `<https://static.docs.arm.com/den0018/a/DEN0018A_neon_programmers_guide_en.pdf>`_

    Guide for Armv7 NEON programming. Note suitable for Armv8.

