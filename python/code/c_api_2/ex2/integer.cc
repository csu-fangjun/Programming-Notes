#include "Python.h"

#ifdef NDEBUG
#undef NDEBUG
#include <assert.h>
#define NDEBUG
#endif

static PyObject *Construct(PyObject *self, PyObject *args) {
  PyObject *p = PyLong_FromLong(1000); // return a new reference

  assert(PyLong_Check(p));            // every type has PyXXX_Check()
  assert(PyLong_CheckExact(p));       // PyXXX_CheckExact()
  assert(Py_REFCNT(p) == 1);          // get reference count
  assert(Py_TYPE(p) == &PyLong_Type); // get type

  assert(PyLong_AsLong(p) == 1000);

  Py_XINCREF(p);
  assert(Py_REFCNT(p) == 2); // get reference count

  Py_XDECREF(p);
  Py_XDECREF(p); // now it is freed

  // PyObject* PyLong_FromUnsignedLong(unsigned long v)¶
  //  return a new reference
  p = PyLong_FromUnsignedLong(-1);
  assert(PyLong_AsUnsignedLong(p) == (unsigned long)-1);
  Py_XDECREF(p);

  // PyObject* PyLong_FromString(const char *str, char **pend, int base)¶
  // return a new reference
  p = PyLong_FromString("123", nullptr, 0);
  assert(PyLong_AsLong(p) == 123);
  Py_XDECREF(p);

  // PyObject* PyLong_FromDouble(double v)
  // return a new reference
  p = PyLong_FromDouble(1.9);
  assert(PyLong_AsLong(p) == 1);
  Py_XDECREF(p);

  int a = 10;
  // PyObject* PyLong_FromVoidPtr(void *p)
  // return a new reference
  p = PyLong_FromVoidPtr(&a);
  assert(PyLong_AsVoidPtr(p) == &a);
  assert(*(int *)PyLong_AsVoidPtr(p) == a);

  Py_RETURN_NONE;
}

static PyMethodDef integer_methods[] = {
    {"construct", Construct, METH_VARARGS, "Constructor"},
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef integermodule = {
    PyModuleDef_HEAD_INIT,
    "integer",        // module name
    "doc of integer", // module doc
    -1,
    integer_methods,
};

PyMODINIT_FUNC PyInit_integer() {
  PyObject *m;
  m = PyModule_Create(&integermodule);
  if (m == nullptr)
    return nullptr;

  return m;
}
