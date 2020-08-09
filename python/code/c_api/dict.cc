#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cassert>
#include <iostream>

static void New() {
  PyObject* dict = PyDict_New();
  assert(Py_REFCNT(dict) == 1);
  assert(PyDict_Check(dict) == 1);

  Py_XDECREF(dict);
}

static void SetItem() {
  PyObject* dict = PyDict_New();
  assert(Py_REFCNT(dict) == 1);

  // use string as key
  PyObject* s = PyUnicode_FromString("hello");
  assert(Py_REFCNT(s) == 1);
  PyDict_SetItemString(dict, "foo", s);  // it increases the reference count!
  assert(Py_REFCNT(s) == 2);             // now it is 2

  PyObject* v = PyDict_GetItemString(dict, "foo");  // it borrows the reference
  assert(Py_REFCNT(s) == 2);                        // still 2

  PyObject_SetItem(dict, s, s);
  assert(Py_REFCNT(s) == 4);  // 4

  Py_XDECREF(dict);
  assert(Py_REFCNT(s) == 1);  // now it is 1
  Py_XDECREF(s);
}

void TestDict(PyObject* m) {
  New();
  SetItem();
}
