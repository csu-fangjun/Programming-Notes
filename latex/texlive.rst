texlive
=======

Installation
------------

Refer to `<https://www.tug.org/texlive/acquire-iso.html>`_ to download it.

.. code-block::

  wget https://mirrors.aliyun.com/CTAN/systems/texlive/Images/texlive2020-20200406.iso
  mount -t iso9660 -o ro,loop,noauto /your/texlive.iso /some/local/path

Refer to `<https://www.tug.org/texlive/quickinstall.html>`_ for installation.

If we do not have root permission, use:

.. code-block::

  sudo apt-get install p7zip-full
  7z x ./path/to/texlive.iso

After extraction, run:

.. code-block::

  perl ./install-tl

Press ``D`` to set the installation directory and press ``I`` to start installation.

Add the following to PATH:

.. code-block::

  export PATH=/xxx/software/texlive/2020/bin/x86_64-linux:$PATH


``which xelatex`` should show its path.
