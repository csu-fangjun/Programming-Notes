Internals
=========


indexing in c++
----------------

`<https://pytorch.org/cppdocs/notes/tensor_indexing.html>`_


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

cuda
----

- ``git checkout v0.1.1``

In `csrc/cuda/Module.cpp`:

  - To create a tuple::

    PyObject *args = PyTuple_New(0);
    bool args_ok = PyTuple_Size(args) == 1;
    Py_DECREF(args);

  - It wraps some common methods: ``cudaSetDevice``, ``cudaGetDevice``, ``cudaGetDeviceCount``

  - How to extend the module ``torch.cuda``?

    (1) Get the module object ``torch.cuda``

    (2) Get its ``__dict__`` object

    (3) Extend ``__dict__``

    .. code-block:: c++

      PyObject *torch_module = PyImport_ImportModule("torch.cuda");
      if (!torch_module) {
        THPUtils_setError("class loader couldn't access torch module");
        return NULL;
      }
      PyObject* module_dict = PyModule_GetDict(torch_module);
      return PyBool_FromLong(THCPModule_initCuda(module_dict));

      //
      ASSERT_TRUE(PyDict_SetItemString(module_dict, "hasMagma", PyBool_FromLong(true)) != -1);


How to define a new type?
-------------------------

Refer to ``torch/csrc/generator.{h,cpp}``.

1. The C struct looks like:

   .. code-block:: cpp

    struct THPGenerator {
      PyObject_HEAD
      THGenerator *cdata;
    };

2. Define a constructor for it:

   .. code-block:: cpp

    static PyObject * THPGenerator_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
    {
      HANDLE_TH_ERRORS
      if ((args && PyTuple_Size(args) != 0) || kwargs) {
        THPUtils_setError("torch.Generator doesn't constructor doesn't accept any arguments");
        return NULL;
      }
      THPGeneratorPtr self = (THPGenerator *)type->tp_alloc(type, 0);
      self->cdata = THGenerator_new();

      return (PyObject*)self.release();
      END_HANDLE_TH_ERRORS
    }

  The function name is irrelevant, but the type and the number of arguments matter. Basically,
  it uses ``tp_alloc`` to allocate memory and then set its struct members accordingly.

3. Define a destructor for it:

   .. code-block:: cpp

      static void THPGenerator_dealloc(THPGenerator* self)
      {
        THGenerator_free(self->cdata);
        Py_TYPE(self)->tp_free((PyObject*)self);
      }

  Inside the destructor, it first frees any memory associated with its members that are allocated in
  the constructor. After that, it uses ``tp_free`` to free the memory of this object.

4. Define a type for it:

   0 in ``PyVarObject_HEAD_INIT(NULL, 0)`` means variable size is 0, which is the common
   case.

   ``tp_basicsize`` is the size of the struct, which is used in ``tp_alloc``.

   ``tp_itemsize`` is 0 since it is neither a list nor a tuple.

   We only need to set its ``tp_dealloc`` and ``tp_new`` (besides ``tp_flags``).


   .. code-block:: cpp

      extern PyObject *THPGeneratorClass;

      PyTypeObject THPGeneratorType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "torch.C.Generator",                   /* tp_name */
        sizeof(THPGenerator),                   /* tp_basicsize */
        0,                                     /* tp_itemsize */
        (destructor)THPGenerator_dealloc,     /* tp_dealloc */
        0,                                     /* tp_print */
        0,                                     /* tp_getattr */
        0,                                     /* tp_setattr */
        0,                                     /* tp_reserved */
        0,                                     /* tp_repr */
        0,                                     /* tp_as_number */
        0,                                     /* tp_as_sequence */
        0,                                     /* tp_as_mapping */
        0,                                     /* tp_hash  */
        0,                                     /* tp_call */
        0,                                     /* tp_str */
        0,                                     /* tp_getattro */
        0,                                     /* tp_setattro */
        0,                                     /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
        NULL,                                  /* tp_doc */
        0,                                     /* tp_traverse */
        0,                                     /* tp_clear */
        0,                                     /* tp_richcompare */
        0,                                     /* tp_weaklistoffset */
        0,                                     /* tp_iter */
        0,                                     /* tp_iternext */
        0,   /* will be assigned in init */    /* tp_methods */
        0,   /* will be assigned in init */    /* tp_members */
        0,                                     /* tp_getset */
        0,                                     /* tp_base */
        0,                                     /* tp_dict */
        0,                                     /* tp_descr_get */
        0,                                     /* tp_descr_set */
        0,                                     /* tp_dictoffset */
        0,                                     /* tp_init */
        0,                                     /* tp_alloc */
        THPGenerator_pynew,                    /* tp_new */
      };

5. Define a function to initialize it.

   What the initialization does it to associate this new type with a module.

   Inside the initialization func, it calls ``PyType_Ready`` for the type,
   increase its reference count, and assign it as an attribute of a module.

   .. code-block:: cpp

      bool THPGenerator_init(PyObject *module)
      {
        THPGeneratorClass = (PyObject*)&THPGeneratorType;
        if (PyType_Ready(&THPGeneratorType) < 0)
          return false;
        Py_INCREF(&THPGeneratorType);
        PyModule_AddObject(module, "Generator", (PyObject *)&THPGeneratorType);
        return true;
      }

6. Define some helper functions for the type.

   .. code-block:: cpp

    bool THPGenerator_Check(PyObject *obj)
    {
      return Py_TYPE(obj) == &THPGeneratorType;
    }

    PyObject * THPGenerator_newObject()
    {
      // TODO: error checking
      THPObjectPtr args = PyTuple_New(0); // NOTE(fangjun): Memory leak!
      return PyObject_Call((PyObject*)&THPGeneratorType, args, NULL);
    }

  Sometimes it is helpful to define a is subclass function:

  .. code-block:: cpp

    bool THPStorage_(IsSubclass)(PyObject *storage)
    {
      return PyObject_IsSubclass((PyObject*)Py_TYPE(storage), (PyObject*)&THPStorageType);
    }

  It uses ``PyObject_IsSubclass``.

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
