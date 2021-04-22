Useful Functions
================

topk
----

See
  - `<https://pytorch.org/docs/stable/generated/torch.topk.html>`_
  - `<https://pytorch.org/docs/stable/tensors.html#torch.Tensor.topk>`_

.. literalinclude:: ./code/useful_functions/topk.py
  :caption: code/useful_functions/topk.py
  :language: python
  :linenos:

cat
---

See

  - `<https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat>`_

.. literalinclude:: ./code/useful_functions/cat.py
  :caption: code/useful_functions/cat.py
  :language: python
  :linenos:

stack
-----

See

  - `<https://pytorch.org/docs/stable/generated/torch.stack.html>`_

Note all input tensors must have the same shape.

.. literalinclude:: ./code/useful_functions/stack.py
  :caption: code/useful_functions/stack.py
  :language: python
  :linenos:

narrow
------

See

  - `<https://pytorch.org/docs/stable/generated/torch.narrow.html#torch.narrow>`_
  - `<https://pytorch.org/docs/stable/tensors.html#torch.Tensor.narrow>`_
  - `<https://pytorch.org/docs/stable/tensors.html#torch.Tensor.narrow_copy>`_

.. literalinclude:: ./code/useful_functions/narrow.py
  :caption: code/useful_functions/narrow.py
  :language: python
  :linenos:

chunk
-----

See

  - `<https://pytorch.org/docs/master/generated/torch.chunk.html#torch.chunk>`_

.. literalinclude:: ./code/useful_functions/chunk.py
  :caption: code/useful_functions/chunk.py
  :language: python
  :linenos:

pad
---

See

- `<https://pytorch.org/docs/stable/nn.functional.html>`_

.. literalinclude:: ./code/useful_functions/pad.py
  :caption: code/useful_functions/pad.py
  :language: python
  :linenos:

as_tensor
---------

See

- `<https://pytorch.org/docs/stable/generated/torch.as_tensor.html#torch.as_tensor>`_

It is used in `<https://github.com/pytorch/pytorch/blob/master/torch/utils/data/sampler.py#L180>`_.

.. literalinclude:: ./code/useful_functions/as_tensor.py
  :caption: code/useful_functions/as_tensor.py
  :language: python
  :linenos:

torch.Generator
---------------

See

- `<https://pytorch.org/docs/stable/generated/torch.Generator.html>`_

It is used in `<https://github.com/pytorch/pytorch/blob/master/torch/utils/data/sampler.py#L114>`_.

Note its ``seed()`` method also changes self.

.. literalinclude:: ./code/useful_functions/generator.py
  :caption: code/useful_functions/generator.py
  :language: python
  :linenos:

randint
-------

See

- `<https://pytorch.org/docs/stable/generated/torch.randint.html#torch.randint>`_

.. literalinclude:: ./code/useful_functions/randint.py
  :caption: code/useful_functions/randint.py
  :language: python
  :linenos:

nn.Sequential
-------------

See

- `<https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_

Its constructor accepts ``*args``, which can either be an OrderedDict
or an iterable of nn.Modules.

See `torch/nn/modules/container.py`.


.. literalinclude:: ./code/useful_functions/sequential.py
  :caption: code/useful_functions/sequential.py
  :language: python
  :linenos:
