2021-03-10
----------

TODOs
~~~~~

1. k2/csrc/ragged_test.cu, remove line line 2017 in `PrefixTest`. That variable is never
   used and is copied from a previous code block.

2. What are the new features in C++17?

    `<https://stackoverflow.com/questions/38060436/what-are-the-new-features-in-c17>`_

3. Changes between C++14 and C++17 DIS

    `<https://isocpp.org/files/papers/p0636r0.html>`_

4. Structured bindings

    `<http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0144r0.pdf>`_

    `<http://open-std.org/JTC1/SC22/WG21/docs/papers/2016/p0144r2.pdf>`_

2021-01-27
----------

TODOs
~~~~~

- Cooperative Groups: Flexible CUDA Thread Programming

    `<https://developer.nvidia.com/blog/cooperative-groups/>`_

2021-01-22
----------

TODOs
~~~~~

- An Introduction to Arithmetic Coding

    `<https://www.cs.cmu.edu/~aarti/Class/10704/Intro_Arith_coding.pdf>`_, a tutorial paper.

2021-01-21
----------

TODOs
~~~~~

- Discriminative Training of Hidden Markov Models

    `<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.9409&rep=rep1&type=pdf>`_

    A Ph.D thesis.

- Large scale discriminative training for speech recognition

    `<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.399.7723&rep=rep1&type=pdf>`_

    A paper by Dan in 2000.

- Minimum Bayes-risk automatic speech recognition

    `<https://sci-hub.se/10.1006/csla.2000.0138>`_, a paper in 2000.

- Machine Learning Paradigms for Speech Recognition: An Overview

    `<http://cvsp.cs.ntua.gr/courses/patrec/slides_material2018/slides-2018/DengLi_MLParadigms-SpeechRecogn-AnOverview_TALSP13.pdf>`_


2021-01-15
----------

TODOs
~~~~~

- The Python 2.3 Method Resolution Order

    `<https://www.python.org/download/releases/2.3/mro/>`_

- `<https://docs.python.org/3/library/functions.html#property>`_

    How to define a property in Python?

2021-01-14
----------

TODOs
~~~~~

- `<https://github.com/tiny-dnn/tiny-dnn>`_

2021-01-13
----------

- PEP 353 -- Using ssize_t as the index type

    `<https://www.python.org/dev/peps/pep-0353/>`_

TODOs
~~~~~

- Parsing arguments and building values

    `<https://docs.python.org/3/c-api/arg.html>`_

- PEP 384 -- Defining a Stable ABI

    `<https://www.python.org/dev/peps/pep-0384/>`_.

    It lists a subset of APIs.


2021-01-09
----------

TODOs
~~~~~

- PEP 465 -- A dedicated infix operator for matrix multiplication

    `<https://legacy.python.org/dev/peps/pep-0465/>`_

2021-01-08
----------

TODOs
~~~~~

- Statistical Language Models Based on Neural Networks

    PhD thesis: `<https://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf>`_

2021-01-07
----------

- Scalable parallel programming with CUDA

    `<https://dl.acm.org/doi/pdf/10.1145/1365490.1365500>`_

    A magazine paper.

TODOs
~~~~~

- Course on CUDA Programming on NVIDIA GPUs, July 22-26, 2019

  `<https://people.maths.ox.ac.uk/gilesm/cuda/>`_

- Efficient Estimation of Word Representations in Vector Space

    This is the paper for Word2Vec.

    `<https://pub-tools-public-publication-data.storage.googleapis.com/pdf/41224.pdf>`_

- Distributed representations of words and phrases and their compositionality

    `<https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf>`_

- A Neural Probabilistic Language Model

    `<http://www-labs.iro.umontreal.ca/~felipe/IFT6010-Automne2011/resources/tp3/bengio03a.pdf>`_


2020-12-30
----------

KenLM
~~~~~

- `<https://zhuanlan.zhihu.com/p/63884335>`_

    A step by step guide illustrating how to compute the probabilities inside kenLM.

- N-gram Language Models

    `<https://web.stanford.edu/~jurafsky/slp3/3.pdf>`_
    from the book ``Speech and Language Processing``.

- Language Modelling

    `<http://www.statmt.org/mtm12/pub/lm.pdf>`_, course slides.

    The last page describes how arpa file works!



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
          // ...
        endif()

    - Linux::

        if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
          // ...
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
