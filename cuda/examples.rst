Examples
========


Example 1
----------

.. literalinclude:: ./code/0-hello-world.cu
  :caption: code/0-hello-world.cu
  :language: cpp
  :linenos:


Example 2
---------

This examples show how to use various IDs inside the kernel.

It is an 1-D example. Inside the kernel,

  - number of blocks is specified by ``gridDim.x``
  - number of threads per block is specified by ``blockDim.x``
  - thread ID inside a block is ``threadIdx.x``
  - number of total threads is ``gridDim.x * blockDim.x``
  - There is no ``threadDim.x`` !!!

.. literalinclude:: ./code/1-hello.cu
  :caption: code/1-hello.cu
  :language: cpp
  :linenos:


Example 3
---------

.. literalinclude:: ./code/2-vector-add.cu
  :caption: code/2-vector-add.cu
  :language: cpp
  :linenos:
