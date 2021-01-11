
C API
=====

1. how to define a module
2. how to assign attributes to a module

   - list
   - tuple
   - dict
   - callable

3. how to define a new class


``python3-config --prefix`` prints ``/usr``, so
``Python.h`` is inside ``/usr/include/python3.7/Python.h``.

References:

- Python/C API Reference Manual

    `<https://docs.python.org/3.9/c-api/index.html>`_

- `<https://tech.blog.aknin.name/2010/04/02/pythons-innards-introduction/>`_

    It introduces ``dis.dis`` and ``Python/ceval.c``

- Type Objects

    `<https://docs.python.org/3/c-api/typeobj.html#type-objects>`_

    It describes `PyTypeObject`

- Descriptor HowTo Guide

    `<https://docs.python.org/3/howto/descriptor.html>`_

- Inside The Python Virtual Machine

    `<https://leanpub.com/insidethepythonvirtualmachine/read>`_


dis
---

Example code:

.. code-block::

  def add(a, b):
    return a + b
  import dis
  print(dis.dis(add))

It prints::

  1           0 LOAD_FAST                0 (a)
              3 LOAD_FAST                1 (b)
              6 BINARY_ADD
              7 RETURN_VALUE

In ``Python/ceval.c``::

        case TARGET(BINARY_ADD): {
            PyObject *right = POP();
            PyObject *left = TOP();
            PyObject *sum;
            if (PyUnicode_CheckExact(left) &&
                     PyUnicode_CheckExact(right)) {
                sum = unicode_concatenate(tstate, left, right, f, next_instr);
                /* unicode_concatenate consumed the ref to left */
            }
            else {
                sum = PyNumber_Add(left, right);
                Py_DECREF(left);
            }
            Py_DECREF(right);
            SET_TOP(sum);
            if (sum == NULL)
                goto error;
            DISPATCH();
        }


Example code 2::

  def sub(a, b):
    return a - b
  import dis
  print(dis.dis(sub))

It prints::

  1           0 LOAD_FAST                0 (a)
              3 LOAD_FAST                1 (b)
              6 BINARY_SUBTRACT
              7 RETURN_VALUE

In ``Python/ceval.c``::

        case TARGET(BINARY_SUBTRACT): {
            PyObject *right = POP();
            PyObject *left = TOP();
            PyObject *diff = PyNumber_Subtract(left, right);
            Py_DECREF(right);
            Py_DECREF(left);
            SET_TOP(diff);
            if (diff == NULL)
                goto error;
            DISPATCH();
        }

Example code 3::

  def item(a, i):
    return a[i]

  dis.dis(item)

It prints::

  1           0 LOAD_FAST                0 (a)
              3 LOAD_FAST                1 (i)
              6 BINARY_SUBSCR
              7 RETURN_VALUE

In ``Python/ceval.c``::

        case TARGET(BINARY_SUBSCR): {
            PyObject *sub = POP();
            PyObject *container = TOP();
            PyObject *res = PyObject_GetItem(container, sub);
            Py_DECREF(container);
            Py_DECREF(sub);
            SET_TOP(res);
            if (res == NULL)
                goto error;
            DISPATCH();
        }

``def attr(a): return a.b`` translates to::

  1           0 LOAD_FAST                0 (a)
              3 LOAD_ATTR                0 (b)
              6 RETURN_VALUE

        case TARGET(LOAD_ATTR): {
            PyObject *name = GETITEM(names, oparg);
            PyObject *owner = TOP();
            PyObject *res = PyObject_GetAttr(owner, name);
            Py_DECREF(owner);
            SET_TOP(res);
            if (res == NULL)
                goto error;
            DISPATCH();
        }







object.h
--------

Refer to `<https://github.com/python/cpython/blob/master/Include/object.h>`_.

.. code-block::

  #define PyObject_HEAD                   PyObject ob_base;

  #define PyObject_HEAD_INIT(type)        \
      { _PyObject_EXTRA_INIT              \
          1, type },

  #define PyVarObject_HEAD_INIT(type, size)       \
        { PyObject_HEAD_INIT(type) size },

    typedef struct _object {
      _PyObject_HEAD_EXTRA
      Py_ssize_t ob_refcnt;
      PyTypeObject *ob_type;
    } PyObject;

    typedef struct {
      PyObject ob_base;
      Py_ssize_t ob_size; /* Number of items in variable part */
    } PyVarObject;

    #define _PyObject_CAST(op) ((PyObject*)(op))
    #define _PyVarObject_CAST(op) ((PyVarObject*)(op))
    #define Py_REFCNT(ob)           (_PyObject_CAST(ob)->ob_refcnt)
    #define Py_TYPE(ob)             (_PyObject_CAST(ob)->ob_type)
    #define Py_SIZE(ob)             (_PyVarObject_CAST(ob)->ob_size)

    static inline int _Py_IS_TYPE(const PyObject *ob, const PyTypeObject *type) {
        return ob->ob_type == type;
    }
    #define Py_IS_TYPE(ob, type) _Py_IS_TYPE(_PyObject_CAST_CONST(ob), type)

    static inline void _Py_SET_REFCNT(PyObject *ob, Py_ssize_t refcnt) {
        ob->ob_refcnt = refcnt;
    }
    #define Py_SET_REFCNT(ob, refcnt) _Py_SET_REFCNT(_PyObject_CAST(ob), refcnt)

    static inline void _Py_SET_TYPE(PyObject *ob, PyTypeObject *type) {
        ob->ob_type = type;
    }
    #define Py_SET_TYPE(ob, type) _Py_SET_TYPE(_PyObject_CAST(ob), type)

    static inline void _Py_SET_SIZE(PyVarObject *ob, Py_ssize_t size) {
        ob->ob_size = size;
    }
    #define Py_SET_SIZE(ob, size) _Py_SET_SIZE(_PyVarObject_CAST(ob), size)


_Py_IDENTIFIER
--------------

``Include/cpython/object.h``

.. code-block::

    typedef struct _Py_Identifier {
        struct _Py_Identifier *next;
        const char* string;
        PyObject *object;
    } _Py_Identifier;

    #define _Py_static_string_init(value) { .next = NULL, .string = value, .object = NULL }
    #define _Py_static_string(varname, value)  static _Py_Identifier varname = _Py_static_string_init(value)
    #define _Py_IDENTIFIER(varname) _Py_static_string(PyId_##varname, #varname)

Example usage:

.. code-block::

  _Py_IDENTIFIER(__doc__);

it Creates::

  static _Py_Identifier _Py_Identifier PyId___doc__ = {.next = NULL, .string = "__doc__", .object = NULL};

PyTypeObject
------------

- ``Include/object.h``

.. code-block::

    /* PyTypeObject structure is defined in cpython/object.h.
       In Py_LIMITED_API, PyTypeObject is an opaque structure. */
    typedef struct _typeobject PyTypeObject;

- ``Include/cpython/object.h``

.. code-block::

    struct _typeobject {
        PyObject_VAR_HEAD
        const char *tp_name; /* For printing, in format "<module>.<name>" */
        Py_ssize_t tp_basicsize, tp_itemsize; /* For allocation */

PyType_Type
-----------

- ``Objects/typeobject.c``

.. code-block::

    PyTypeObject PyType_Type = {
        PyVarObject_HEAD_INIT(&PyType_Type, 0)
        "type",                                     /* tp_name */
        sizeof(PyHeapTypeObject),                   /* tp_basicsize */
        sizeof(PyMemberDef),                        /* tp_itemsize */
        (destructor)type_dealloc,                   /* tp_dealloc */
        offsetof(PyTypeObject, tp_vectorcall),      /* tp_vectorcall_offset */
        0,                                          /* tp_getattr */
        0,                                          /* tp_setattr */
        0,                                          /* tp_as_async */
        (reprfunc)type_repr,                        /* tp_repr */
        0,                                          /* tp_as_number */
        0,                                          /* tp_as_sequence */
        0,                                          /* tp_as_mapping */
        0,                                          /* tp_hash */
        (ternaryfunc)type_call,                     /* tp_call */
        0,                                          /* tp_str */
        (getattrofunc)type_getattro,                /* tp_getattro */
        (setattrofunc)type_setattro,                /* tp_setattro */
        0,                                          /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
        Py_TPFLAGS_BASETYPE | Py_TPFLAGS_TYPE_SUBCLASS |
        Py_TPFLAGS_HAVE_VECTORCALL,                 /* tp_flags */
        type_doc,                                   /* tp_doc */
        (traverseproc)type_traverse,                /* tp_traverse */
        (inquiry)type_clear,                        /* tp_clear */
        0,                                          /* tp_richcompare */
        offsetof(PyTypeObject, tp_weaklist),        /* tp_weaklistoffset */
        0,                                          /* tp_iter */
        0,                                          /* tp_iternext */
        type_methods,                               /* tp_methods */
        type_members,                               /* tp_members */
        type_getsets,                               /* tp_getset */
        0,                                          /* tp_base */
        0,                                          /* tp_dict */
        0,                                          /* tp_descr_get */
        0,                                          /* tp_descr_set */
        offsetof(PyTypeObject, tp_dict),            /* tp_dictoffset */
        type_init,                                  /* tp_init */
        0,                                          /* tp_alloc */
        type_new,                                   /* tp_new */
        PyObject_GC_Del,                            /* tp_free */
        (inquiry)type_is_gc,                        /* tp_is_gc */
    };

PyModule_Create
----------------

- ``Include/modsupport.h``

.. code-block::

    /* The PYTHON_ABI_VERSION is introduced in PEP 384. For the lifetime of
       Python 3, it will stay at the value of 3; changes to the limited API
       must be performed in a strictly backwards-compatible manner. */
    #define PYTHON_ABI_VERSION 3
    #define PYTHON_ABI_STRING "3"

    #define PyModule_Create(module) \
            PyModule_Create2(module, PYTHON_ABI_VERSION)

PyModule_Create2
----------------

- ``Objects/object.c``

PyErr_WarnFormat
----------------

.. code-block::

    int err;
    err = PyErr_WarnFormat(PyExc_RuntimeWarning, 1,
        "Python C API version mismatch for module %.100s: "
        "This Python has API version %d, module %.100s has version %d.",
         name,
         PYTHON_API_VERSION, name, module_api_version);
    if (err)
        return 0;

PyErr_Format
------------

.. code-block::

    if (module->m_slots) {
        PyErr_Format(
            PyExc_SystemError,
            "module %s: PyModule_Create is incompatible with m_slots", name);
        return NULL;
    }

PyErr_SetString
---------------

.. code-block::

  PyErr_SetString(PyExc_SystemError, "nameless module");

PyUnicode_FromString
--------------------

.. code-block::

    PyObject *
    PyModule_New(const char *name)
    {
        PyObject *nameobj, *module;
        nameobj = PyUnicode_FromString(name);
        if (nameobj == NULL)
            return NULL;
        module = PyModule_NewObject(nameobj);
        Py_DECREF(nameobj);
        return module;
    }

Py_None
-------

.. code-block::

    PyAPI_DATA(PyObject) _Py_NoneStruct; /* Don't use this directly */
    #define Py_None (&_Py_NoneStruct)

    /* Macro for returning Py_None from a function */
    #define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None

PyObject_Str
------------

- ``Include/object.c``

.. code-block::

  PyObject *
  PyObject_Str(PyObject *v)
  {
      PyObject *res;
      if (v == NULL)
          return PyUnicode_FromString("<NULL>");
      if (PyUnicode_CheckExact(v)) {
          Py_INCREF(v);
          return v;
      }

      if (Py_TYPE(v)->tp_str == NULL)
          return PyObject_Repr(v);

      res = (*Py_TYPE(v)->tp_str)(v);
      if (res == NULL) {
          return NULL;
      }

      if (!PyUnicode_Check(res)) {
          _PyErr_Format(tstate, PyExc_TypeError,
                        "__str__ returned non-string (type %.200s)",
                        Py_TYPE(res)->tp_name);
          Py_DECREF(res);
          return NULL;
      }

      if (PyUnicode_READY(res) < 0) {
          return NULL;
      }

      return res;
  }

PyObject_Repr
-------------

.. code-block::

  PyObject *
  PyObject_Repr(PyObject *v)
  {
      PyObject *res;

      if (v == NULL)
          return PyUnicode_FromString("<NULL>");
      if (Py_TYPE(v)->tp_repr == NULL)
          return PyUnicode_FromFormat("<%s object at %p>",
                                      Py_TYPE(v)->tp_name, v);

      res = (*Py_TYPE(v)->tp_repr)(v);

      if (res == NULL) {
          return NULL;

      if (!PyUnicode_Check(res)) {
          _PyErr_Format(tstate, PyExc_TypeError,
                        "__repr__ returned non-string (type %.200s)",
                        Py_TYPE(res)->tp_name);
          Py_DECREF(res);
          return NULL;
      }

      if (PyUnicode_READY(res) < 0) {
          return NULL;
      }
      return res;
  }

Exceptions
----------

Refer to `<https://docs.python.org/3/c-api/exceptions.html>`_

Relevant files are:
- ``Include/pyerrors.h``
- ``Python/errors.c``
- ``Include/warnings.h``
- ``Python/_warnings.c``
