
C API LongObject
================

- ``Include/longintrepr.h``

.. code-block::

    /* Long integer representation.
       The absolute value of a number is equal to
            SUM(for i=0 through abs(ob_size)-1) ob_digit[i] * 2**(SHIFT*i)
       Negative numbers are represented with ob_size < 0;
       zero is represented by ob_size == 0.
       In a normalized number, ob_digit[abs(ob_size)-1] (the most significant
       digit) is never zero.  Also, in all cases, for all valid i,
            0 <= ob_digit[i] <= MASK.
       The allocation function takes care of allocating extra memory
       so that ob_digit[0] ... ob_digit[abs(ob_size)-1] are actually available.

       CAUTION:  Generic code manipulating subtypes of PyVarObject has to
       aware that ints abuse  ob_size's sign bit.
    */

    struct _longobject {
        PyObject_VAR_HEAD
        digit ob_digit[1];
    };

- ``Include/longobject.h``

.. code-block::

  typedef struct _longobject PyLongObject; /* Revealed in longintrepr.h */

PyLong_Type
-----------

- ``Objects/longobject.c``

The ``tp_name`` is ``int``, so ``type(1)`` prints ``int`` in python.

``tp_basicsize`` is the size in bytes we should allocoate while creating the object.

``tp_itemsize`` is the size in bytes per item in the variable length part.

.. code-block::

    PyTypeObject PyLong_Type = {
        PyVarObject_HEAD_INIT(&PyType_Type, 0)
        "int",                                      /* tp_name */
        offsetof(PyLongObject, ob_digit),           /* tp_basicsize */
        sizeof(digit),                              /* tp_itemsize */
        0,                                          /* tp_dealloc */
        0,                                          /* tp_vectorcall_offset */
        0,                                          /* tp_getattr */
        0,                                          /* tp_setattr */
        0,                                          /* tp_as_async */
        long_to_decimal_string,                     /* tp_repr */
        &long_as_number,                            /* tp_as_number */
        0,                                          /* tp_as_sequence */
        0,                                          /* tp_as_mapping */
        (hashfunc)long_hash,                        /* tp_hash */
        0,                                          /* tp_call */
        0,                                          /* tp_str */
        PyObject_GenericGetAttr,                    /* tp_getattro */
        0,                                          /* tp_setattro */
        0,                                          /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
            Py_TPFLAGS_LONG_SUBCLASS,               /* tp_flags */
        long_doc,                                   /* tp_doc */
        0,                                          /* tp_traverse */
        0,                                          /* tp_clear */
        long_richcompare,                           /* tp_richcompare */
        0,                                          /* tp_weaklistoffset */
        0,                                          /* tp_iter */
        0,                                          /* tp_iternext */
        long_methods,                               /* tp_methods */
        0,                                          /* tp_members */
        long_getset,                                /* tp_getset */
        0,                                          /* tp_base */
        0,                                          /* tp_dict */
        0,                                          /* tp_descr_get */
        0,                                          /* tp_descr_set */
        0,                                          /* tp_dictoffset */
        0,                                          /* tp_init */
        0,                                          /* tp_alloc */
        long_new,                                   /* tp_new */
        PyObject_Del,                               /* tp_free */
    };

_PyLong_New
------------

- ``Objects/longobject.c``

.. code-block::

    PyLongObject *
    _PyLong_New(Py_ssize_t size) {

        PyLongObject *result;
        result = PyObject_MALLOC(offsetof(PyLongObject, ob_digit) +
                                 size*sizeof(digit));
        return (PyLongObject*)PyObject_INIT_VAR(result, &PyLong_Type, size);
    }

- ``Include/cpython/objimpl.h``

.. code-block::

    #define PyObject_INIT_VAR(op, typeobj, size) \
        _PyObject_INIT_VAR(_PyVarObject_CAST(op), (typeobj), (size))

    static inline PyVarObject*
    _PyObject_INIT_VAR(PyVarObject *op, PyTypeObject *typeobj, Py_ssize_t size)
    {
        assert(op != NULL);
        Py_SET_SIZE(op, size);
        PyObject_INIT((PyObject *)op, typeobj);
        return op;
    }

    #define PyObject_INIT(op, typeobj) \
        _PyObject_INIT(_PyObject_CAST(op), (typeobj))

    static inline PyObject*
    _PyObject_INIT(PyObject *op, PyTypeObject *typeobj)
    {
        assert(op != NULL);
        Py_SET_TYPE(op, typeobj);
        if (PyType_GetFlags(typeobj) & Py_TPFLAGS_HEAPTYPE) {
            Py_INCREF(typeobj);
        }
        _Py_NewReference(op);
        return op;
    }

-  ``Objects/object.c``

.. code-block::

    void
    _Py_NewReference(PyObject *op)
    {
        Py_SET_REFCNT(op, 1);
    }

long_to_decimal_string
----------------------

``longobject`` does not have ``tp_str``.
Its ``tp_repr`` is ``long_to_decimal_string``.
