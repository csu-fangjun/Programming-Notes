#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cassert>

/*
// clang-format off
relevant files:

Include/cpython/listobject.h
Include/listobject.h
Objects/listobject.c
Objects/clinic/listobject.c.h

PyTypeObject PyList_Type = {
  PyVarObject_HEAD_INIT(&PyType_Type, 0)
  "list", // tp_name
  sizeof(PyListObject), // tp_basicsize
  0, // tp_itemsize
};

typedef struct {
  PyObject_VAR_HEAD // ;
  PyObject** ob_item;
  Py_sssize_t allocated;
} PyListObject;

PyList_New (Py_ssize_t size):
  PyListObject* op = PyObject_GC_New(PyListObject, &PyList_Type);
  op->ob_item = (PyObject**)PyMem_Calloc(size, sizeof(PyObject*));
  Py_SET_SIZE(op, size);
  op->allocated = size;
  _PyObject_GC_TRACK(op);
  return (PyObject*)op;

// clang-format off
*/

static void New() {
  PyObject* list = PyList_New(2);

  assert(PyList_Check(list) == 1);     // it is a list
  assert(Py_REFCNT(list) == 1);        // reference count is init to 1
  assert(PyList_Size(list) == 2);      // there are two items
  assert(PyList_GET_SIZE(list) == 2);  // without error checking

  assert(PyList_GetItem(list, 0) == NULL);  // items are initialized to NULL
  assert(PyList_GetItem(list, 1) == NULL);

  Py_XDECREF(list);
}

static void SetItem() {
  // note that it uses list->ob_item[i] = value;
  // so it steals the reference!
  // the caller does **NOT** own the value after calling SetItem()!
  PyObject* list = PyList_New(2);

  PyObject* s = PyUnicode_FromString("hello");
  assert(Py_REFCNT(s) == 1);

  PyList_SetItem(list, 0, s);  // it steals the reference!
  // inside the list, item[0] = s; this is how it steals the reference
  assert(Py_REFCNT(s) == 1);  // still 1!

  // at this point, we should not use s any more
  // since we do not own it!

  PyObject* v = PyList_GetItem(list, 0);
  assert(v == s);             // it returns item[0], which is s itself!
                              // so we can use address comparison here
  assert(Py_REFCNT(v) == 1);  // it is still 1!

  Py_XINCREF(Py_None);
  PyList_SetItem(list, 0, Py_None);  // it destroys the previous item[0]!

  // v and s are destroyed!

  Py_XDECREF(list);
}

static void InsertItem() {
  // It increases the reference!
  PyObject* list = PyList_New(2);
  assert(Py_REFCNT(list) == 1);
  assert(PyList_Size(list) == 2);

  PyObject* s = PyUnicode_FromString("hello");
  assert(Py_REFCNT(s) == 1);

  PyList_Insert(list, 0, s);       // in creases the reference count of s
  assert(PyList_Size(list) == 3);  // the list is resized!
  assert(Py_REFCNT(s) == 2);       // reference count is 2!

  PyObject* v = PyList_GetItem(list, 0);
  assert(v == s);

  Py_XDECREF(list);  // this frees items in list

  assert(Py_REFCNT(s) == 1);  // not it is 1
  Py_XDECREF(s);              // free s
}

static void AppendItem() {
  // it increases the reference!
  PyObject* list = PyList_New(2);
  assert(Py_REFCNT(list) == 1);
  assert(PyList_Size(list) == 2);

  PyObject* s = PyUnicode_FromString("hello");
  assert(Py_REFCNT(s) == 1);

  PyList_Append(list, s);          // in creases the reference count of s
  assert(PyList_Size(list) == 3);  // the list is resized!
  assert(Py_REFCNT(s) == 2);       // reference count is 2!

  PyObject* v = PyList_GetItem(list, 2);  // get the last item
  assert(v == s);

  Py_XDECREF(list);  // this frees items in list

  assert(Py_REFCNT(s) == 1);  // not it is 1
  Py_XDECREF(s);              // free s
}

void TestList(PyObject* m) {
  New();
  SetItem();
  InsertItem();
  AppendItem();
}
