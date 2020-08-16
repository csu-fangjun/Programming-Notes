
PCRE
====

The latest release is ``PCRE2``.

Its official website is `<https://www.pcre.org/>`_.


PCRE1
-----
.. code-block::

   wget https://ftp.pcre.org/pub/pcre/pcre-8.44.zip
   unzip pcre-8.44.zip
   cd pcre-8.44

   # read README in the project for compiling.
   ./configure --prefix=$HOME/software/pcre
   make
   make install

After installation,
- ``bin`` contains an executable ``pcre-config``, which is similar to ``pkg-config`` and ``python3-config``
- ``include`` contains the header files
- ``lib`` contains the libraries.
- ``lib/pkgconfig`` contains ``*.pc`` files for ``pkg-config``.


PCRE2
-----
.. code-block::

   wget https://ftp.pcre.org/pub/pcre/pcre2-10.35.zip
   unzip pcre2-10.35.zip
