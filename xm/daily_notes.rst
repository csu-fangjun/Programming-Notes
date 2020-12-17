
2020-12-12
----------

- What every systems programmer should know about concurrency

    `<https://assets.bitbashing.io/papers/concurrency-primer.pdf>`_

- `<https://en.wikipedia.org/wiki/Test-and-set>`_

    Test-and-Set Lock is short for TSL.

    See test-test-and-set.

    See `<https://en.cppreference.com/w/cpp/atomic/atomic_flag>`_.

- `<https://en.wikipedia.org/wiki/Compare-and-swap>`_

    Compare and Swap is short for CAS.

- A Simple GPU Hash Table

    `<https://nosferalatu.com/SimpleGPUHashTable.html>`_


TODO
~~~~

- What is warp divergence in CUDA?

2020-12-08
----------

- google/benchmark

    The first commit is 403f3544 on 2013.12.19

    - `CMAKE_CXX_FLAGS`: `-Wall -Werror -std=c++0x`
    - `CMAKE_CXX_FLAGS_DEBUG`: `-g -O0 -DDEBUG`
    - `CMAKE_CXX_FLAGS_RELEASE`: `-fno-strict-aliasing -O3 -DNDEBUG`

    To detect for different operating systems in CMake:

    - macOS::

      if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        ...
      endif()

    - Linux::

      if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        ...
      endif()

    - Windows::

      if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

    - To detect x86 CPU::

      if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86")
        add_definitions(-DARCH_X86)
      endif()
        ...
      endif()

    This is how it defines ``arraysize``::

        template <typename T, size_t N>
        char (&ArraySizeHelper(T (&array)[N]))[N];

        #define arraysize(array) (sizeof(ArraySizeHelper(array)))

    It uses some tricks to define the macro ``STATIC_ASSERT``. It also
    defines ``CHECK``, ``CHECK_EQ``, ``CHECK_NE`` and so on.

- `<https://github.com/google/nvidia_libs_test>`_

    benchmark of cuDNN with google benchmark.

    It also uses abseil!

- `<https://docs.nvidia.com/cuda/cuda-samples/index.html#simple>`_

    CUDA samples

2020-12-08
----------

- Read source code of PyTorch

    - git reset --hard  c7d7d # initial revamp of torch7 tree


2020-12-05
----------

- How to Implement Performance Metrics in CUDA C/C++

    `<https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/>`_

    It describes how to do timing using CUDA event and how to measure bandwidth.

- How to Query Device Properties and Handle Errors in CUDA C/C++

    `<https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/>`_

- How to Optimize Data Transfers in CUDA C/C++

    `<https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/>`_

    Measure bandwidth of paged locked memory.

- PinnedMemoryAllocator in PyTorch

    aten/src/ATen/cuda/PinnedMemoryAllocator.{h,cpp}
    aten/src/THC/THCGeneral.cpp
    aten/src/THC/THCCachingHostAllocator.h


- `<https://on-demand.gputechconf.com/gtc/2014/presentations/S4158-cuda-streams-best-practices-common-pitfalls.pdf>`_



2020-11-27
----------

TODO
~~~~

- `<https://github.com/pytorch/audio/blob/fb3ef9ba427acd7db3084f988ab55169fab14854/packaging/pkg_helpers.bash#L123>`_
  says it uses soumith/manylinux-cuda* Docker image.

    The problem is how to build k2 with manylinux wheels?

- smoke test

    There is a folder in torch/audio: `<https://github.com/pytorch/audio/tree/master/.circleci/smoke_test/docker>`_.

    Refer to wikipedia for what the meaning of smoke test is. It lists a reference book::

      Lessons Learned in Software Testing: A Context-Driven Approach

    The rating of the book on Amazon is 4.6/5, and 8.4/10.0 on douban. It can be downloaded
    from `<http://gen.lib.rus.ec/>`_.
