Code notes
==========

The first commits uses OpenFST 1.2.7

To compile openfst 1.2.7, use:

.. code-block::

    cd kaldi/tools
    wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.2.7.tar.gz
    tar xvf openfst-1.2.7.tar.gz
    ln -s openfst-1.2.7/src openfst
    cd openfst-1.2.7
    ./configure LDFLAGS=-Wl,--no-as-needed --prefix=$PWD --enable-static --disable-shared

Compile kaldi:

.. code-block::

    cd kaldi/src
    ./configure --with-math-lib=MKL --mkl-root=/opt/intel/mkl
