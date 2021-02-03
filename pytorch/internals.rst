Internals
=========

Sparse tensors
--------------


- Added sparse support for asin and neg functions, updated log1p

    `<https://github.com/pytorch/pytorch/pull/44028>`_

- Roadmap for PyTorch Sparse Tensors

    `<https://github.com/Quansight-Labs/rfcs/tree/pearu/rfc0005/RFC0003-sparse-roadmap>`_

- The state of PyTorch tensor layouts support

    `<https://github.com/Quansight-Labs/rfcs/blob/pearu/rfc0005/RFC0003-sparse-roadmap/SparseSupportState.md>`

- torch.sparse improvements - tracking issue

    `<https://github.com/pytorch/pytorch/issues/44634>`_

- Sparse-sparse matrix multiplication (CPU/CUDA)

    `<https://github.com/pytorch/pytorch/pull/39526>`_

    It lists two papers:

      - Sparse GPU Kernels for Deep Learning
      - The State of Sparsity in Deep Neural Networks

- Efficient Storage Scheme for n-Dimensional Sparse Array: GCRS/GCCS

  A paper: `<https://www.researchgate.net/profile/Md_Abu_Hanif_Shaikh/publication/312167966_Efficient_storage_scheme_for_n-dimensional_sparse_array_GCRSGCCS/links/5874260708aebf17d3b0cf47.pdf>`_



autograd
--------

- `<https://github.com/pytorch/pytorch/commit/53f00ae429aa1bd18b407ffd17d06c9e85578edf#diff-743abcafd32ad0e69f39ac5a91df4197b7e1921c135cacee7ef6dc829a8a7af8>`_

    The first commit that added autograd to PyTorch, in pure Python.

    The committer is `Adam Paszke <https://github.com/apaszke>`_. He also wrote a paper
    `Automatic differentiation in PyTorch <https://openreview.net/pdf?id=BJJsrmfCZ>`_
    and the corresponding slide is `<https://autodiff-workshop.github.io/slides/Paszke_ad_in_pytorch.pdf>`_.



alpha release
-------------

`<https://github.com/pytorch/pytorch/releases?after=v0.1.11>`_


- alpha-1 release v0.1.1 (2016.09.01): `<https://github.com/pytorch/pytorch/archive/v0.1.1.tar.gz>`_
- alpha-2 release v0.1.2 (2016.09.01): `<https://github.com/pytorch/pytorch/archive/v0.1.2.tar.gz>`_

Issues
------

- `<https://github.com/pytorch/pytorch/issues/3>`_, 2016.08.17

    Discussed about Python code style: PEP 8.

- `<https://github.com/pytorch/pytorch/issues/5>`_, 2016.08.27

    The plan for the first public release


2016-06-25: 2b53cce
-------------------

PyTuple
~~~~~~~

.. code-block::

  PyObject `*`args = PyTuple_New(0);

BuildValue
~~~~~~~~~~

.. code-block::

  PyObject `*`kwargs = Py_BuildValue("{s:N}", "cdata", PyLong_FromVoidPtr(ptr));

where ``N`` means a ``PyObject*``. See `<https://docs.python.org/3/c-api/arg.html>`_.

IsSubClass
~~~~~~~~~~

.. code-block::

  bool THPStorage_(IsSubclass)(PyObject `*`storage)
  {
    return PyObject_IsSubclass((PyObject*)Py_TYPE(storage), (PyObject*)&THPStorageType);
  }

``PyObject_IsSubclass(derived, cls)``, see `<https://docs.python.org/3/c-api/object.html>`_.

dealloc
~~~~~~~

It is assigned to ``tp_dealloc``.

.. code-block::

  static void THPStorage_(dealloc)(THPStorage* self)
  {
    THStorage_(free)(self->cdata);
    Py_TYPE(self)->tp_free((PyObject*)self);
  }

tpnew
~~~~~

.. code-block::

  static PyObject * THPStorage_(pynew)(PyTypeObject `*`type, PyObject `*`args, PyObject `*`kwargs)

It invokes ``tp_alloc(type, 0)`` to allocate memory; 0 means the memory size is from ``tp_basicsize``.

bool
~~~~

.. code-block::

    return PyBool_FromLong(true);

References
----------

- `<https://pytorch.org/blog/a-tour-of-pytorch-internals-1/>`_
- `<https://pytorch.org/blog/a-tour-of-pytorch-internals-2/>`_

- A quick tour of Torch internals

  `<https://apaszke.github.io/torch-internals.html>`

  It's about Torch, not PyTorch, but it is still informative. There is a hot
  discussion of why not to switch to C++.
