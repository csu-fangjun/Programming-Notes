#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cassert>

/*
// clang-format off
relevant files:

Include/cpython/tupleobject.h
Include/tupleobject.h
Objects/tupleobject.c
Objects/clinic/tuplebject.c.h

PyTypeObject PyTuple_Type = {
  PyVarObject_HEAD_INIT(&PyType_Type, 0)
  "tuple", // tp_name
  sizeof(PyTupleObject) - sizeof(PyObject*), // tp_basicsize
  sizeof(PyObject*), // tp_itemsize
};

typedef struct {
  PyObject_VAR_HEAD
  PyObject* ob_item[1];
} PyTupleObject;

PyTuple_New (Py_ssize_t size):
  PyTupleObject* op = PyObject_GC_NewVar(PyTupleObject, &PyTuple_Type, size);
  for i in to to size:
    op->obitem[i] = NULL
  _PyObject_GC_TRACK(op);
  return (PyObject*)op;

// clang-format off
*/

static void New() {
  PyObject* tuple = PyTuple_New(2);
  assert(PyTuple_Check(tuple) == 1);
  assert(PyTuple_Size(tuple) == 2);
  assert(Py_REFCNT(tuple) == 1);

  Py_XDECREF(tuple);
}

static void SetItem() {
  // note that it **steals** the reference.
  // The caller does not own the reference after
  // calling SetItem() !
  PyObject* tuple = PyTuple_New(2);
  PyObject* s = PyUnicode_FromString("hello");
  assert(Py_REFCNT(s) == 1);

  PyTuple_SetItem(tuple, 0, s);  // steals the reference
  assert(Py_REFCNT(s) == 1);

  PyObject* v = PyTuple_GetItem(tuple, 0);  // borrows the reference
  assert(v == s);

  Py_XDECREF(tuple);

  // now it's invalid to access s, since the underlying object
  // has been destroyed.
}

void TestTuple(PyObject* m) {
  New();
  SetItem();
  // there is no InsertItem and AppendItem for tuple!
}
