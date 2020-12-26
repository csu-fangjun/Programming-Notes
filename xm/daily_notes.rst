
2020-12-25
----------

Wav2Letter
~~~~~~~~~~

1. It proposed `ASG`, automiatic segmentation criteria

There is a blog
`<https://mobilemonitoringsolutions.com/presentation-wav2letter-facebooks-fast-open-source-speech-recognition-system/>`_
about it. It mentions several c++ library:

  - ArrayFire, an open-source tensor library, supporting CPU, CUDA and OpenCL
  - Flashlight, a neural network library, built on top of ArrayFire

    - It has autograd!

`<https://github.com/facebookresearch/wav2letter>`_

`<https://github.com/facebookresearch/wav2letter/wiki/Data-Preparation>`_
describes the data format expected by wave2letter.

Letter-based speech recognition with gated convnets `<https://arxiv.org/pdf/1712.09444.pdf>`_
says that ASG without transitions are hard to train.

Wav2Letter++: The fastest open-source speech recognition system
`<https://arxiv.org/pdf/1812.07625.pdf>`_

`<https://github.com/facebookresearch/wav2letter/tree/v0.2/tutorials/1-librispeech_clean>`_

  tutorial about wav2letter with librispeech


The librispeech dataset is in `/home/storage04/zhuangweiji/data/open-source-data/librispeech/`.

TODO
~~~~

- Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data

    Cited more that 14784 times!

    `<https://nlp.cs.nyu.edu/nycnlp/lafferty01conditional.pdf>`_

    `<https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Conditional+Random+Fields%3A+Probabilistic+Models+for+Segmenting+and+Labeling+Sequence+Dat&btnG=>`_

- `<https://github.com/kpu/kenlm>`_

    For language modeling. Read its code!



2020-12-24
----------

TODOs
~~~~~

1. `<https://github.com/k2-fsa/k2/pull/427#discussion_r547625364>`_

    Resolve this comments!

2020-12-22
----------

TODOs
~~~~~

- 1. ragged_ops.cu, line 198, in RaggedShapeFromTotSizes

    Allocate a big block of memory.

- 2 . ragged_ops.cu

    Context() can be assigned to a reference, for example, in GetRowInfo.

    Read the implementation of `AppendAxis0()`. How to use TaskRedirect?


2020-12-18
----------

Compilation of torchaudio::

  sudo apt-get install libsox-dev
  python setup.py bdist_wheel

Notes about torchaudio code
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**backend**

  ``backend/utils.py``

  ``torchaudio.info``: ``info`` is an attribute of ``torchaudio``,
  which is set in ``backend/utils.py``. It is called by ``utils._init_audio_backend()``
  and ``_init_audio_backed`` is in ``backend/__init__.py`` and is called automatically
  on import.

  There are other three methods like ``info``: ``save``, ``load``, ``load_wav``.

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
