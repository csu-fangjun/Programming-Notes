Dataset
=======

See ``pytorch/torch/utils/data/dataset.py``.

Sampler
-------

See ``pytorch/torch/utils/data/sampler.py``.

The base class ``Sampler`` provides nothing.

The ``SequentialSampler`` has two methods: ``__iter__``
and ``__len__``. Its ``__iter__`` returns ``iter(range(len(self.data_source)))``.
That is why it is called **sequential** sampler. Its ``__iter__`` returns a ``iterator``.

The ``RandomSamplers`` uses ``yield from`` in its ``__iter__``.
Read coroutines in ``xm/todo.rst`` to understand it. It supports
``replacement`` sampling. Read its ``__iter__``. Somethings to take away:

  - Use ``torch.randint`` to generate a list of random integers and use ``yield from``
    to return it. The ``__iter__`` method returns a generator!
  - Use ``torch.randperm`` to generate a permuted list of integers. It also uses ``yield from``.

``RandomSamplers.__iter__`` returns a generator.
