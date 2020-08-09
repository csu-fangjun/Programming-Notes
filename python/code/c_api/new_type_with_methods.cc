#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cassert>
#include <cstring>
#include <iostream>

// clang-format off
struct People{
		PyObject_HEAD
		PyObject* first; // first name
		PyObject* last; // last name
};
// clang-format on

// PyType_GenericNew calls type->tp_alloc to allocate memory
// for the struct object; set the ob_type and the reference count.
// Everything else is uninitialized.
//
// The default tp_alloc is `PyType_GenericAlloc`

static PyObject* People_new(PyTypeObject* type, PyObject* args,
                            PyObject* kwargs) {
  // the default tp_new is NULL, which can be set to PyType_GenericNew
  People* self = (People*)(type->tp_alloc(type, 0));
  std::cout << "in new: " << self << "\n";
  assert(self != nullptr);

  self->first = PyUnicode_FromString("hello first");
  self->last = PyUnicode_FromString("hello second");
  return (PyObject*)self;
}

// typedef void(*destructor)(PyObject*); in Include/object.h
static void People_dealloc(People* self) {
  // the default tp_alloc calls tp_free to free the object itself.
  // we have to manually free its members
  Py_XDECREF(self->first);
  Py_XDECREF(self->last);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// typedef int(*initproc)(PyObject*, PyObject*, PyObject*);
static int People_init(People* self, PyObject* args, PyObject* kwargs) {
  std::cout << "in init: " << self << "\n";
  assert(args == nullptr || PyTuple_Check(args) == 1);
  assert(kwargs == NULL || PyDict_Check(kwargs) == 1);

  // it has to be char*, not const char*!
  static char* kwlist[] = {"first", "last", nullptr};
  PyObject* first = nullptr;
  PyObject* last = nullptr;
  std::cout << "here" << std::endl;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", kwlist, &first,
                                   &last)) {
    // "|" means options after it are optional
    return -1;  // -1 for error
  }

  if (first) {
    // we have the first argument
    PyObject* tmp = self->first;
    Py_XINCREF(first);
    self->first = first;
    // Py_XDECREF(tmp);
  }

  if (last) {
    // we have the first argument
    PyObject* tmp = self->last;
    Py_XINCREF(last);
    self->last = last;
    // Py_XDECREF(tmp);
  }
  return 0;  // 0 for success
}

// add method to Type
static PyObject* People_name(People* self, PyObject* Py_UNUSED(ignored)) {
  if (self->first == nullptr) {
    PyErr_SetString(PyExc_AttributeError, "failed to get People.first");
    return nullptr;
  }

  if (self->last == nullptr) {
    PyErr_SetString(PyExc_AttributeError, "failed to get People.second");
    return nullptr;
  }

  return PyUnicode_FromFormat("%S %S", self->first, self->last);
}

static PyObject* People_set_first(People* self, PyObject* args) {
  assert(args != nullptr);
  assert(PyTuple_Check(args) == 1);  // args is a tuple
  PyObject* first;
  if (!PyArg_ParseTuple(args, "O", &first)) {
    // it returns false, so the exception has already been set
    assert(PyErr_Occurred() != nullptr);
    // PyErr_SetString(PyExc_RuntimeError, "invalid argument");
    return nullptr;
  }
  std::cout << "reference count: " << Py_REFCNT(first) << "\n";

  if (first) {
    PyObject* tmp = self->first;
    Py_XINCREF(first);
    self->first = first;
    Py_XDECREF(tmp);
  }

  Py_RETURN_NONE;
}

static PyMethodDef People_methods[] = {
    {"name", (PyCFunction)People_name, METH_NOARGS,
     "return the name of the People"},
    {"set_first", (PyCFunction)People_set_first, METH_VARARGS,
     "set the first name"},
    {nullptr},
};

static void AddNewTypeWithMethods(PyObject* m) {
  // TODO(fangjun): use Python allocator to create a new type object!
  PyTypeObject* PeopleType = new PyTypeObject;
  std::memset(PeopleType, 0, sizeof(PyTypeObject));
  Py_XINCREF(PeopleType);

  PeopleType->tp_name = "hello.People";
  PeopleType->tp_doc = "doc of hello.People";  // optional
  PeopleType->tp_basicsize = sizeof(People);
  PeopleType->tp_itemsize = 0;
  PeopleType->tp_flags = Py_TPFLAGS_DEFAULT;

  PeopleType->tp_new = People_new;
  PeopleType->tp_dealloc = (destructor)People_dealloc;
  PeopleType->tp_methods = People_methods;
  PeopleType->tp_init = (initproc)People_init;

  if (PyType_Ready(PeopleType) < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot create type PeopleType");
    return;
  }

  if (PyModule_AddObject(m, "People", (PyObject*)PeopleType) < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot add PeopleType");
    delete PeopleType;
    return;
  }
}

void TestNewTypeWithMethods(PyObject* m) { AddNewTypeWithMethods(m); }
