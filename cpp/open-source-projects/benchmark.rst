
Benchmark
=========

- GitHub: `<https://github.com/google/benchmark>`_

Installation
------------

.. code-block:: bash

  git clone --depth 1 https://github.com/google/benchmark.git
  cd benchmark
  mkdir build
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_TESTING=OFF
  make -j10

The generated libraries are in the directory ``build/src``.

To build a program with benchmark, use:

- compiler flag: ``-Iinclude -std=c++11``
- linker flag ``-Lbuild/src -lbenchmark``

.. WARNING::

  For gcc, we have to add ``-pthread`` to the linker flag.
  Note that it is NOT ``-lpthread``.

Example
-------

.. literalinclude:: ./code/benchmark/hello.cc
   :caption: hello.cc
   :language: cpp
   :linenos:

.. literalinclude:: ./code/benchmark/Makefile
   :caption: Makefile
   :language: makefile
   :linenos:
