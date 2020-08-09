

Day 0
=====

.. code-block::

   sudo apt-get install qemu

The boot sector has 512 bytes, whose last two bytes has to be 0x55 and 0xaa.
That is, byte 510 is ``0x55`` and byte 511 is ``0xaa``.


.. literalinclude:: ./code/day0/boot.s
  :caption: boot.s
  :language: asm
  :linenos:

.. literalinclude:: ./code/day0/Makefile
  :caption: Makefile
  :language: makefile
  :linenos:


