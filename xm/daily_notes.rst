
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
