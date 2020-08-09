
A64 Assembly
============


Registers
---------

General registers:

  - r0-r30. We cannot use ``rx`` directly in code. The size of the register has to be specified.
    For instance to use ``r0`` as 32-bit, we use ``w0``; to use it as 64-bit, we use ``x0``.

    .. NOTE::

      ``w0`` and ``x0`` refer to the same register. ``w0`` is the lower 32-bit and ``x0`` is the
      whole 64-bit.

  - r31. It can be used either as a zero register, ``wzr`` (32-bit) or ``xzr`` (64-bit), or
    used as a stack pointer ``sp`` (64-bit), ``wsp`` (32-bit).

     .. NOTE::

        ``sp`` **MUST** be 16-byte aligned; otherwise, a stack alignment exception will be generated.

FP/SIMD Registers:

  - v0-v31. There are thirty-two 128-bit registers.

    .. WARNING::

      ``v0`` and ``x0`` are not the same register!

    Usage (scalar register):

      - ``b0``, the lower 8-bit of ``v0``, higher bits are ginored on read and are set to zero on write
      - ``h0``, the lower 16-bit of ``v0``
      - ``s0``, the lower 32-bit of ``v0``
      - ``d0``, the lower 64-bit of ``v0``
      - ``v0``

    Usage (vecotr register):

      - ``v0.8b``, 8-bit x 8-lane, the upper 64-bit is ignored on read and is set to zero on write
      - ``v0.16b``, 8-bit x 16-lane
      - ``v0.4h``, 16-bit x 4-lane, the upper 64-bit is igored on read and is set to zero on write
      - ``v0.8h``, 16-bit x 8-lane
      - ``v0.2s``, 32-bit x 2-lane, the uppser 64-bit is ignored on read and is set to zero on write
      - ``v0.4s``, 32-bit x 4-lane

    Usage (vecotr element):

      - ``v0.b[0]``, bit 0-7, same as ``v0.8b[0]`` and ``v0.16b[0]``
      - ``v0.b[1]``, bit 8-15
      - ``v0.s[1]``, bit 32-63, same as ``v0.2s[1]`` and ``v0.4s[1]``


Address Modes
-------------

  - simple register::

      ldr x0, [x1]        // x0 = *x1; load 8-byte
      ldrsw x0, [x1]      // load 4-byte, then sign-extend it to x0
      ldrsh  x0, [x1]      // load 2-byte, then sign-extend it to x0
      ldrsb x0, [x1]      // load 1-byte, then sign-extend it to x0

      ldr w0, [x1]        // load 4-byte from x1

      ldrb w0, [x1]       // load 1-byte, then zero-extend to w0
      ldrsb w0, [x1]      // load 1-byte, then sign-extend it to w0

      ldrh  w0, [x1]      // load 2-byte, then zero-extend it to w0
      ldrsh  w0, [x1]      // load 2-byte, then sign-extend it to w0

      ldrh  x0, [x1]      // load 2-byte, then zero-extend it to x0

      str w0, [x1]        // save 4-byte to memory
      str x0, [x1]        // save 8-byte to memory
      strb w0, [x1]       // save 1-byte to memory
      strhw w0, [x1]       // save 2-byte to memory

  - offset::

      ldr x0, [x1, 10]            // x0 = *(x1 + 10)

      ldr x0, [x1, x2]            // x0 = *(x1 + x2)
      ldr x0, [x1, x2, lsl 3]     // x0 = *(x1 + (x2 << 3))

  - post-indexed::

      ldr x0, [x1], 10    // x0 = *x1; x1 += 10;

  - pre-indexed::

      ldr x0, [x1, 10]!   // x0 = *(x1 + 10); x1 += 10;


Arithmetics
-----------

.. code-block::

  add x0, x1, x2            // x0 = x1 + x2
  add x0, x1, 3             // x0 = x1 + 3
  add x0, x1, x2, lsl 3     // x0 = x1 + (x2 << 3)
  add x0, x1, x2, lsr 2     // x0 = x1 + (x2 >> 2)

  sub x0, x1, x2            // x0 = x1 - x2
  neg x0, x1                // x0 = -x1
  and x0, x1, x2            // x0 = x1 & x2
  orr x0, x1, x2            // x0 = x1 | x2
  err x0, x1, x2            // x0 = x1 ^ x2
  bic x0, x1, x2            // x0 = x1 & (~x2)  bit-wise clear

  lsl x0, x1, x2            // x0 = x1 << x2
  lsl x0, x1, 2             // x0 = x1 << 2
  lsr x0, x1, 3             // x0 = x1 >> 3

  mul x0, x1, x2            // x0 = x1 * x2
  sdiv x0, x1, x2           // x0 = x1 / x2   signed division
  usdiv x0, x1, x2          // x0 = x1 / x2   unsigned division


Branches
--------

- ``eq`` (==), ``ne`` (!=),
- ``lt`` (<), ``gt`` (>)
- ``le`` (<=), ``ge`` (>=)


Example::

    cmp x0, x1
    b.eq 2f
    1:
    add x1, x2, 1
    b 3f
    2:
    add x1, x2, 2
    3:

- ``cbz x0, 1f``

    jump to label ``1`` if ``x0 == 0``

    compare and branch if zero

- ``cbnz x0, 1f``

    jump to label ``1`` if ``x0 != 0``

    compare and branch if not zero

Stack
-----

The stack grows in the same direction as ``x86``, i.e., towards higher addresses.
It is **full-descending**.

Different from ``x86``, when we call a function, the stack pointer is not changed!
The return address is saved in a default register, i.e., ``x30``, so there is no
need to change the stack pointer ``sp``.

.. code-block::

  sub sp, sp, 64              // reserve 64 bytes stack memory
  str x19, [sp, 0]            // save x19 to sp[0]
  stp x20, x21, [sp, 8]       // save x20 to sp[8], save x21 to sp[16]

  ldr x19, [sp, 0]            // load x19 from sp[0]
  ldp x20, x21, [sp, 8]       // load x20 from sp[8], load x21 from sp[16]
  add sp, sp, 64              // free 64 bytes stack memory

.. WARNING::

  ``sp`` **MUST** be a multiple of 16. That is, ``sp`` has to be 16-byte aligned.
  Otherwise, an exception will throw at runtime.

References
----------

- Programmer’s Guide for ARMv8-A Version: 1.0
  `<https://static.docs.arm.com/den0024/a/DEN0024A_v8_architecture_PG.pdf>`_

    A **very good** short manual about ARMv8-A programming.

- http://infocenter.arm.com/help/index.jsp

    It provides a search box to search help info for instructions.

- Arm® A64 Instruction Set Architecture Armv8, for Armv8-A architecture profile
  `<https://static.docs.arm.com/ddi0596/b/DDI_0596_ARM_a64_instruction_set_architecture.pdf>`_

    A single PDF contains all the instructions of Armv8.

- ARMv8 Instruction Set Overview `<https://www.element14.com/community/servlet/JiveServlet/previewBody/41836-102-1-229511/ARM.Reference_Manual.pdf>`_

    On page 8, it says:

      The A64 assembly language does not require the ‘#’ symbol to introduce immediate values, though an assembler
      must allow it. An A64 disassembler shall always output a ‘#’ before an immediate value for readability.

- ARM64 Assembly Language Notes `<http://cit.dixie.edu/cs/2810/arm64-assembly.pdf>`_

    It gives a brief overview of A64 assembly language. The part for passing
    extra arguments to function calls is worth reading.

- Computer Organization and Design ARM Edition: The Hardware Software Interface
  `<https://www.amazon.com/Computer-Organization-Design-ARM-Architecture-dp-0128017333/dp/0128017333/>`_

    A classicial book! This ARM edition is specific to ARMv8.

    An electronic edition can be found at GitHub address:
    `<https://github.com/AbderrhmanAbdellatif/ComputerOrganization>`_

- arm64 assembly crash course
  `<https://github.com/Siguza/ios-resources/blob/master/bits/arm64.md>`_

    Some notes for A64 assembly.

- Procedure Call Standard for the Arm® 64-bit Architecture (AArch64)
  `<https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst>`_

    The latest documentation for aarch64 calling conventions.

    It also describes the implementation of ``va_args``!

- C++ ABI for the Arm 64-bit Architecture (AArch64)
  `<https://static.docs.arm.com/ihi0059/d/cppabi64.pdf>`_

- https://github.com/ARM-software/optimized-routines

    Optimized implementations of various library functions for ARM architecture processors

    A good place for learning A64 assembly programming, especially about the string operations
    `<https://github.com/ARM-software/optimized-routines/tree/master/string/aarch64>`_

- Modern Assembly Language Programming with the ARM Processor Suggested Laboratory Exercises

    `<http://www.mcs.sdsmt.edu/lpyeatt/courses/314/labs.pdf>`_

    October 31, 2016, by Larry D. Pyeatt

    This is about arm32, but it can be used to learn arm64

- https://github.com/torvalds/linux/blob/master/include/uapi/asm-generic/unistd.h

    arm64 sys call number

- http://www.mcs.sdsmt.edu/lpyeatt/courses/314/

    Assembly Language course for ARM32
