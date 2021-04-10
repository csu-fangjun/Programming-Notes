
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


