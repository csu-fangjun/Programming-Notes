
ctags
=====

Install from source
-------------------

.. code-block::

    git clone https://github.com/universal-ctags/ctags.git
    cd ctags
    ./autogen.sh
    ./configure --prefix=$HOME/software/ctags
    make
    make install

It produces two executables: ``ctags/bin/ctags`` and ``ctags/bin/readtags``.

Use ctags with vim
------------------

1. Generate ``tags``. Go to the project root directory and run::

      ctags -R .

which generates a text file ``tags``.

