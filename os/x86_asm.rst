
x86 asm
========

addressing
----------

.. code-block::

  mov %eax, var(,1)     # *(var + 1) = eax
  move (%ebx), %eax     # eax = *ebx
  mov -4(%esi), %eax    # eax = *(esi - 4)
  mov %cl, (%esi, %eax, 1)  # *(esi + eax*1) = cl
  mov (%esi, %ebx, 4), %edx # edx = *(esi + ebx * 4)

  lea (%ebx, %esi, 8), %edi   # edi = ebx + esi * 8

stack
-----

.. code-block::

  push %eax   #  esp = esp - 4;  *esp = eax
  pop %edi    # edi = *esp; esp = esp + 4



.. literalinclude:: ./code/x86_asm/datatype.c
  :caption: datatype.c
  :language: c
  :linenos:

.. literalinclude:: ./code/x86_asm/datatype.s
  :caption: datatype.s
  :language: asm
  :linenos:

.. literalinclude:: ./code/x86_asm/Makefile
  :caption: Makefile
  :language: makefile
  :linenos:

References
----------

- `<http://flint.cs.yale.edu/cs421/papers/x86-asm/asm.html>`_
