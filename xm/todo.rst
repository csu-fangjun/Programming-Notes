
TODOs
=====

- Read CS 61: Systems Programming and Machine Organization (2018)

    `<https://cs61.seas.harvard.edu/site/2018/>`_

    and do its exercises.

- Read CS360 -- Systems Programming

    `<http://web.eecs.utk.edu/~huangj/cs360/>`_

    and do its exercise.


- Secure Programming for Linux and Unix HOWTO

  `<https://dwheeler.com/>`_

  There are lots of links contained in this persons' home page, include a book:

  `<https://dwheeler.com/secure-programs/Secure-Programs-HOWTO.pdf>`_ (194 pages)

- Read the paper:

  Deep Learning on Mobile Devices – A Review `<https://arxiv.org/pdf/1904.09274.pdf>`_

- `<https://github.com/alibaba/MNN>`_
  and its paper `<https://arxiv.org/pdf/2002.12418.pdf>`_

    MNN: A universal and efficient inference engine

- `<https://github.com/Tencent/TNN>`_ and
  `<https://github.com/Tencent/ncnn>`_

- `<https://github.com/PaddlePaddle/Paddle.git>`_ and
  `<https://github.com/PaddlePaddle/Paddle-Lite>`_


- `<https://github.com/MegEngine/MegRay>`_

    provides a very good example for distributed training!

    Read it!

- `<https://github.com/ARM-software/ComputeLibrary>`_ and
  `<https://github.com/quic/aimet>`_ and
  `<https://github.com/ARM-software/CMSIS_5>`_


- `<https://github.com/zeromq/libzmq>`_

    How is it used for socket programming?

    And its cpp binding: `<https://github.com/zeromq/cppzmq>`_

- `<https://github.com/halide/Halide>`_

    Read it!

    Does it relate to JIT? What is JIT and how to use it?

- Learn autodiff.

  Some related projects are:

    - `<https://github.com/google/jax>`_

    - `<https://github.com/MegEngine/MegEngine>`_

        Learn the third party libraries used in MegEngine.

        Also, there are several blog articles, e.g.,
        `<https://megengine.org.cn/blog/engine-tao-graph-and-matmul-optimization>`_

- Learn how to use SoX.

  How SoX is used in torchaudio?

- Read The Python Standard Library

  `<https://docs.python.org/3/library/index.html>`_

  What are the most useful and commonly used libraries?

- Read the Glossary of Python

    See `<https://docs.python.org/3/glossary.html>`_

- Read the builtin functions of Python and write tests in ``python/code/useful_functions``.

    See `<https://docs.python.org/3/library/functions.html>`_

- How does multiprocessing work in Python?

    For example, how multiprocessing is used in PyTorch's DataLoader when
    its num_workers > 1.

    How the Dataset is split over different workers?

    How do torch.utils.data.get_worker_info() and
    worker_init_func() of torch.utils.data.DataLoader work?

    See pytorch/torch/utils/data/dataset.py

- An overview of gradient descent optimization algorithms

    `<https://ruder.io/optimizing-gradient-descent/>`_ a blog article.

    And its paper: `<https://arxiv.org/pdf/1609.04747.pdf>`_

- `<https://sphinx-gallery.github.io/stable/tutorials/index.html>`_

    Write notebook style documentation for k2!!

    `<https://github.com/pytorch/tutorials/blob/master/intermediate_source/char_rnn_classification_tutorial.py>`_
    is an example and it is rendered as
    `<https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial>`_.

- A guide to recurrent neural networks and backpropagation

  `<https://wiki.eecs.yorku.ca/course_archive/2016-17/F/6327/_media/rn_dallas.pdf>`_.

  backpropagation through time, a detailed guide

- Implement the following and use PyTorch to check it:

    - Linear layer, weight norm
    - RNN, LSTM, GRU
    - Optimizers: SGD, Adam, RMSProp

- Read the code of earlier OpenFST
- Read the code of FST algorithms in k2, especially ``k2.intersect_dense_pruned``

- Read the blog article "Understanding LSTM Networks" `<https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_

Flashlight
----------

- `<https://github.com/facebookresearch/flashlight/blob/master/flashlight/app/asr/augmentation/SoundEffect.cpp>`_

Transformer
-----------

- `<http://nlp.seas.harvard.edu/2018/04/03/attention.html>`_

    A blog article about **Attention is All you Need**


C++
----

- The "Empty Member" C++ Optimization

    `<http://www.cantrip.org/emptyopt.html>`_

- Templates and Inheritance Interacting in C++

  `<https://www.informit.com/articles/article.aspx?p=31473&seqNum=2>`_

- Simple C++11 metaprogramming

  `<https://www.boost.org/doc/libs/1_75_0/libs/mp11/doc/html/simple_cxx11_metaprogramming.html>`_

- Simple C++11 metaprogramming, part 2

  `<https://www.boost.org/doc/libs/1_75_0/libs/mp11/doc/html/simple_cxx11_metaprogramming_2.html>`_

- `<https://github.com/boostorg/mp11>`_

    Mp11, a C++11 metaprogramming library

- Searching for Types in Parameter Packs

  `<http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4115.html>`_
  by Stephan T. Lavavej

- Variable Templates For Type Traits

  `<http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3854.htm>`_

- Template Metaprogramming Part 2

  `<https://nilsdeppe.com/posts/tmpl-part2>`_





Boost
^^^^^

  - `<https://www.boost.org/>`, official website
  - `<https://www.boost.org/doc/libs/1_75_0/>`_, its documentation

  - `<https://theboostcpplibraries.com/>`_, a book

      Learn what boost provides and look into its implementation!

People
------

- `<https://github.com/goldsborough>`_
