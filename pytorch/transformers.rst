
Transformers
============

- ``TORCH_ARG``: defined in ``torch/csrc/api/include/torch/arg.h``.


MultiheadAttentionOptions
-------------------------

See ``torch/csrc/api/include/torch/nn/options/activation.h``, line 586.

and ``torch/csrc/api/src/nn/options/activation.cpp``, line 24.

It sets ``kdim_``, ``vdim_``, and ``embed_dim_`` to ``embed_dim``.

It requires that ``embed_dim_ % num_heads_ == 0``.


MultiheadAttentionImpl
----------------------

``torch/csrc/api/include/torch/nn/modules/activation.h``, line 795
and ``torch/csrc/api/src/nn/modules/activation.cpp``, line 462

``head_dim == embed_dim / num_heads``.

If ``kdim == vdim == embed_dim``, then ``in_proj_weight`` is of shape ``(3 * embed_dim,  embed_dim)``
and ``q_proj_weith``, ``k_proj_weight``, and ``v_proj_weight`` are empty.

Otherwise, ``q_proj_weight`` is of shape ``(embed_dim, embed_dim)``,
``k_proj_weight`` is ``(embed_dim, kdim)``, and ``v_proj_weight`` is ``(embed_dim, vdim)``

``torch/csrc/api/include/torch/nn/functional/activation.h``

``torch/csrc/api/include/torch/nn/functional/activation.h``

``torch/csrc/api/src/nn/modules/activation.cpp``

Python
------

MultiheadAttention
^^^^^^^^^^^^^^^^^^^

- `<https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>`_
- `<https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html>`_

Parameters:

  - ``embed_dim``

      query, key, and value are required to have embed_dim as their last dim
      if ``kdim`` and ``vdim`` is not provided.

      If query, key, and value have the same dim, then there is only one
      input projection weight matrix ``in_proj_weight``, whose shape is
      ``(3 * embed_dim, embed_dim)``.

      ``[0, embed_dim)`` is for query. ``[embed_dim, 2*embed_dim)`` is for key.
      ``[2*embed_dim, 3*embed_dim)`` is for value.

  - ``num_heads``

      ``head_dim * num_heads == embed_dim``

``F.multi_head_attention_forward`` is defined
at `<https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L4633>`_.
