
Basics
======

IDs
---

- ``threadIdx.x``, ``threadIdx.y``, ``threadIdx.z``. Its dimension is
  ``blockDim.x``, ``blockDim.y`` and ``blockDim.z``. That is, the maximum
  ``threadIdx.x`` is ``blockDim.x - 1``

- ``blockIdx.x``, ``blockIdx.y``, ``blockIdx.z``. Maximum ``blockIdx.x``
  is ``gridDim.x``.

Thread bolck
------------

A thread block contains many threads.

.. code-block::

  int ThreadIDs[z][y][x];

The maximum number of threads in a thread block is 1024; that is,
``threadIdx.z * threadIdx.y * threadIdx.x <= 1024``

- For 1-D, thread id is ``x``
- For 2-D, thread is is ``x + y * Dx``
- For 3-D, thread id is ``x + y * Dx + z * Dx * Dy``

Grid block
----------

A grid block contains many thread blocks.

... code-block::

  int BlockIDs[z][y][x];

Kernel invocation
-----------------

.. code-block::

  kernel<<<num_blocks, threads_per_block>>>

  kernel<<<num_blocks, threads_per_block, 0, stream>>>
  // where 0 means shared memory size
  // stream is cudaStream_t


Memory Hierarchy
----------------

- every thread has its own private memory (per thread local memory)
- every block has a shared memory that can be accessed only by threads within this block (per block shared memory).
  Threads in the same block can use ``__syncthreads()`` for synchronization.

- all threads from different blocks can access the ``global`` memory,
  the readonly ``constant`` and ``texture`` memory

Compute Capability
------------------

It is a version number ``major.minor``, also known as SM version.

- 8, Ampere
- 7, Volta, (7.5 is Turing, 2018), 2017
- 6, Pascal, 2016
- 5, Maxwell, 2014
- 3, Kepler, 2012
- 2, Fermi (not supported since CUDA 9.0)
- 1, Tesla, (not supported since CUDA 7.0)

Virtual architecture feature list
---------------------------------

See `<calendar.google.com/calendar/b/1/r/week?pli=1&t=AKUaPmbhSyw15alj0P89WaehlmEd8RCcJBkzVk-KXlrImJ8uRNzQTTRrW4oHrO_3ka65f8Q1rI-haCwz6oFrskFsPxLlRIBE2Q%3D%3D>`_

- ``compute_35``, ``compute_37``, kelper
- ``compute_50``, ``compute_52``, ``compute_53``, Maxwell
- ``compute_60``, ``compute_61``, ``compute_62``, Pascal
- ``compute_70``, ``compute_72``, Volta
- ``compute_75``, Turing
- ``compute_80``, Ampere

The option to specify the virtual gpu is ``-arch`` or
``--gpu-architecture``.

``--gpu-code`` or ``-code`` is for real GPU.

``nvcc x.cu -arch=compute_20 -code=compute_20``: generate PTX for virtual architecture
specified by ``-arch``; generate for for JIT specified by ``-code``.

``nvcc x.cu -arch=compute_30 -code=compute_30,sm_30,sm_35``: generate PTX for virtual
architecture specified by ``-arch``; generate real code for ``sm_30, sm_35`` specified
by ``-code``; generate JIT code for ``compute_30`` specified by ``-code``.

C++ Language Extension
----------------------

__global__
^^^^^^^^^^

- Execute on the device
- Callable from the host
- Callable from the device for devices with compute capability >= 3.2
- It must return void
- It cannot be a class member of a class
- A call to a ``__global__`` function is asynchronous

__device__
^^^^^^^^^^

- Execute on the device
- Callable only from the device

__host__
^^^^^^^^

- Execute on the host
- Callable only from the host
- If no ``__host__``, ``__global__`` and ``__device__`` is given, it is assumed to be ``__host__``
- ``__host__ __global__`` is illegal
- ``__host__ __device__`` is fine

