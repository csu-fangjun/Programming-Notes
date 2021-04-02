
Normalization
=============


Weight Normalization
--------------------

Refer to :cite:`salimans2016weight`.

PyTorch's implementation is available at
`<https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/weight_norm.py>`_.

BatchNormalization
------------------

- Its ``mean`` and ``var`` depend on the batch size
- The training and inference mode are different
- Computation of the ``mean`` and ``var`` are fixed can be applied
  on for a single dimension.

.. literalinclude:: ./code/batch_norm.py
  :caption: batch_norm.py
  :language: python
  :linenos:

LayerNorm
---------

.. literalinclude:: ./code/layer_norm.py
  :caption: layer_norm
  :language: python
  :linenos:

- It has exact behavior for training and inference mode
- Its ``mean`` and ``var`` do not depend on batch size
- The shape for computing ``mean`` and ``var`` is specified by
  the user
