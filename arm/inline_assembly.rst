
GCC Inline Assembly
===================

We can embed inline assembly inside a function. The syntax is as follows::

  asm volatile(
    "instruction 1    \n"
    "instruction 2    \n"
    : // output
    "=r"(variable_name0), // %0
    "=r"(variable_name1)  // %1
    : // input
    "r"(variable_name2)  // %2
    : // clobber list
    "cc", "memory", "x0", "v0"
  );

For more details about the syntax, you can visit `<https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html>`_.

The following are some examples about inline assembly.


move
----

The following code is possibly the simplest inline assembly example
implementing ``b = a``. It demonstrates how to pass c variables to assembly
and to extract the output from assembly.


.. literalinclude:: ./code/inline_asm.cc
   :language: cpp
   :lineno-start: 14
   :lines: 14-23

The disassembler's output is:

.. literalinclude:: ./code/inline_asm.cc
   :language: cpp
   :lineno-start: 39
   :lines: 39-49

I've added some comment about it:

.. literalinclude:: ./code/inline_asm.cc
   :language: cpp
   :lineno-start: 24
   :lines: 24-37

.. HINT::

  This is no need to use ``#`` to introduce an immediate value
  in A64 assembly. For backward compatibility, it is not an error
  to use ``#`` for immediate values.

  The output of a disassembler will **always** use ``#`` for immediate values.

Another example for moving 64-bit integers:

.. literalinclude:: ./code/inline_asm.cc
   :language: cpp
   :lineno-start: 53
   :lines: 53-61

The output of the disassembler is:

.. literalinclude:: ./code/inline_asm.cc
   :language: cpp
   :lineno-start: 80
   :lines: 80-93

My note is:

.. literalinclude:: ./code/inline_asm.cc
   :language: cpp
   :lineno-start: 62
   :lines: 62-78

add
---

The following code show how to use ``add``:

.. literalinclude:: ./code/inline_asm.cc
   :language: cpp
   :lineno-start: 100
   :lines: 100-122

.. NOTE::

  The input operands use ``0``, ``1`` and ``2`` to refer
  the operands from the output. The advantage is that we
  can use less registers in the assembly. The drawback
  is that it requires more ``mov`` instructions.

  From the disassembler's output shown below, we can
  see that the compiler allocates 4 registers instead of
  3 or 6.

The output from the disassembler:

.. literalinclude:: ./code/inline_asm.cc
   :language: cpp
   :lineno-start: 155
   :lines: 155-176

And my comment:

.. literalinclude:: ./code/inline_asm.cc
   :language: cpp
   :lineno-start: 123
   :lines: 123-152
