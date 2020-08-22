

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

Usage of qemu:

.. code-block::

    qemu-system-x86_64 \
      -display curses \
      -monitor telnet:127.0.0.1:1234,server,nowait \
      -drive format=raw,file=boot.bin

    telnet 127.0.0.1 1234

objdump
-------

.. code-block::

  g++ -g -c test.cc
  objdump -S -C test.o

It will show the assembly code for every line of c++ code.

x86 boot process
----------------

After reset, PC is set to ``FFFF:0000``.
The last two bytes of the first sector of the boot disk MUST be 0x55, 0xaa.
The first sector is loaded to 0x7c00.

The first sector is called Master Boot Record (MBR). The progrom in
the first sector is called MBR Bootloader.

This page `<https://wiki.osdev.org/BIOS>`_ lists some commonly used BIOS interrupts number.
BIOS interrupts are only available in **real mode**.

This page `<http://www.ablmcc.edu.hk/~scy/CIT/8086_bios_and_dos_interrupts.htm>`_ gives
a more detailed list.



References
----------

- `<https://0xax.gitbooks.io/linux-insides/content/Booting/linux-bootstrap-1.html>`_

- `<https://wiki.osdev.org/Required_Knowledge>`_

- `<https://github.com/tuhdo/os01>`_

    Operating Systems: From 0 to 1
