
hexedit
=======


Generate a binary file:

.. code-block::

  echo -n -e "\x30\x20\x31\x41" > t.bin

Edit with hexedit
-----------------

.. code-block::

  sudo apt-get install hexedit

  hexedit t.bin

- ``ctrl c``, exit without saving
- ``ctrl x``,  save and exit

Edit with vim
-------------

``vim t.bin`` and run ``:%!xxd``.

It displays:

.. code-block::

  00000000: 0330 2031 410a                           .0 1A.

After editing, use ``:%!xxd -r`` to convert it back.

