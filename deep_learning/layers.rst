
nn.Module in PyTorch
====================

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
  :caption: code/batch_norm.py
  :language: python
  :linenos:

LayerNorm
---------

.. literalinclude:: ./code/layer_norm.py
  :caption: code/layer_norm.py
  :language: python
  :linenos:

- It has exact behavior for training and inference mode
- Its ``mean`` and ``var`` do not depend on batch size
- The shape for computing ``mean`` and ``var`` is specified by
  the user
- Each element of x in (x-mu) has a weight. That is, the weight shape equals
  to that of x.

Linear
------

Note that its weight shape is ``(out_dim, in_dim)``, its input shape
is ``(N, *, in_dim)``, and it uses a transposed ``weigth`` during
computation.

.. literalinclude:: ./code/linear_layer.py
  :caption: code/linear_layer.py
  :language: python
  :linenos:

NLLLOSS
-------
Negative(N) Log (L) Likelihood(L) Loss.

Note that its input is from ``log_softmax``.


.. literalinclude:: ./code/nll_loss.py
  :caption: code/nll_loss.py
  :language: python
  :linenos:
