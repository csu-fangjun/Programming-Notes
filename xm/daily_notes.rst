
2020-11-27
==========

TODO
----

- `<https://github.com/pytorch/audio/blob/fb3ef9ba427acd7db3084f988ab55169fab14854/packaging/pkg_helpers.bash#L123>`_
  says it uses soumith/manylinux-cuda* Docker image.

    The problem is how to build k2 with manylinux wheels?

- smoke test

    There is a folder in torch/audio: `<https://github.com/pytorch/audio/tree/master/.circleci/smoke_test/docker>`_.

    Refer to wikipedia for what the meaning of smoke test is. It lists a reference book::

      Lessons Learned in Software Testing: A Context-Driven Approach

    The rating of the book on Amazon is 4.6/5, and 8.4/10.0 on douban. It can be downloaded
    from `<http://gen.lib.rus.ec/>`_.
