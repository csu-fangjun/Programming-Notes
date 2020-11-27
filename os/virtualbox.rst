
VirtualBox
==========

Add a new disk
--------------

.. code-block::

  sudo fdisk /dev/sdb
  sudo mkfs.ext4 /dev/sdb1
  mkdir shared
  sudo mount /dev/sdb1 ./shared
