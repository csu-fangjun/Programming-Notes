#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cassert>
#include <iostream>

static void SetString() {
  PyErr_SetString(PyExc_RuntimeError, "set string");
  PyObject* v = PyErr_Occurred();  // it returns a borrows reference
  if (v) {
    // error ocurred
    if (PyErr_ExceptionMatches(PyExc_RuntimeError) == 1) {
      // if it is PyExc_RuntimeError, then clear the error
      PyErr_Clear();
    }
  }

  // No need to decrease the reference count of v !
}

static void SetObject() {
  PyObject* val = PyUnicode_FromString("set object");
  assert(Py_REFCNT(val) == 1);

  // note that it increases the reference count!
  PyErr_SetObject(PyExc_RuntimeError, val);
  assert(Py_REFCNT(val) == 2);
  PyErr_Clear();  // it decreases the reference count!
  assert(Py_REFCNT(val) == 1);

  // now for fetch
  PyErr_SetObject(PyExc_RuntimeError, val);
  PyObject* type;
  PyObject* value;
  PyObject* traceback;
  PyErr_Fetch(&type, &value, &traceback);
  assert(PyErr_Occurred() == NULL);

  assert(PyErr_GivenExceptionMatches(type, PyExc_RuntimeError) == 1);
  assert(PyUnicode_Check(value));
  assert(Py_REFCNT(val) == 2);  // since value holds a reference

  PyErr_Restore(type, value, traceback);
  PyErr_Clear();

  Py_XDECREF(val);
}

static void BadArgument() {
  // clang-format off
  // PyErr_SetString(PyExc_TypeError, "bad argument type for built-in operation");
  // clang-format on
  PyErr_BadArgument();
  assert(PyErr_Occurred() != NULL);
  assert(PyErr_ExceptionMatches(PyExc_TypeError) == 1);

  PyObject* type;
  PyObject* value;
  PyObject* traceback;

  PyErr_Fetch(&type, &value, &traceback);
  assert(PyUnicode_Check(value) == 1);
  assert(strcmp(PyUnicode_AsUTF8(value),
                "bad argument type for built-in operation") == 0);
  PyErr_Restore(type, value, traceback);

  PyErr_Clear();
}

static void NoMemory() {
  // PyErr_SetNone(PyExc_MemoryError);
  PyErr_NoMemory();
  assert(PyErr_Occurred() != NULL);
  assert(PyErr_ExceptionMatches(PyExc_MemoryError) == 1);
  PyErr_Clear();
}

static void SetNone() {
  // PyErr_SetObject(<some_exception>, (PyObject*)NULL);
  PyErr_SetNone(PyExc_RuntimeError);
  assert(PyErr_Occurred() != NULL);
  PyObject* type;
  PyObject* value;
  PyObject* traceback;
  PyErr_Fetch(&type, &value, &traceback);
  assert(PyErr_Occurred() == NULL);
  assert(value == NULL);  // note that it is NULL!
  PyErr_Restore(type, value, traceback);
  PyErr_Clear();
}

static void Fetch() {
  PyErr_SetString(PyExc_RuntimeError, "fetch");
  assert(PyErr_Occurred() != NULL);

  PyObject* type;
  PyObject* value;
  PyObject* traceback;
  PyErr_Fetch(&type, &value, &traceback);  // this clears the exception
  assert(PyErr_Occurred() == NULL);

  assert(type != NULL);
  assert(value != NULL);

  assert(PyUnicode_Check(value) == 1);
  assert(strcmp(PyUnicode_AsUTF8(value), "fetch") == 0);
  PyErr_Restore(type, value, traceback);  // this steals the reference

  assert(PyErr_Occurred() != NULL);

  PyErr_Clear();  // clear the exception
  assert(PyErr_Occurred() == NULL);
}

static void Print() {
  PyErr_SetString(PyExc_RuntimeError, "ignore this.");
  assert(PyErr_Occurred() != NULL);
  PyErr_Print();  // it clears the exception and prints the message to stderr
  assert(PyErr_Occurred() == NULL);
}

static void Format() {
  PyErr_Format(PyExc_RuntimeError, "hello %s", "world");
  assert(PyErr_Occurred() != NULL);

  PyObject* type;
  PyObject* value;
  PyObject* traceback;
  PyErr_Fetch(&type, &value, &traceback);
  assert(PyErr_Occurred() == NULL);

  assert(PyErr_GivenExceptionMatches(type, PyExc_RuntimeError) == 1);
  assert(PyUnicode_Check(value) == 1);
  assert(strcmp(PyUnicode_AsUTF8(value), "hello world") == 0);
  PyErr_Restore(type, value, traceback);
  assert(PyErr_Occurred() != NULL);

  PyErr_Clear();
}

static void NewException2() {
  PyObject* ex = PyErr_NewException("hello.MyException", NULL, NULL);

  PyErr_SetString(ex, "my exception");
  assert(PyErr_Occurred() != NULL);
  assert(PyErr_ExceptionMatches(ex) == 1);

  PyErr_Clear();

  Py_XDECREF(ex);
}

static void NewException(PyObject* m) {
  // add a exception to module m
  PyObject* ex = PyErr_NewException("hello.MyException", NULL, NULL);
  assert(PyModule_AddObject(m, "MyException", ex) == 0);
}

void TestException(PyObject* m) {
  assert(PyModule_Check(m) == 1);
  SetString();
  SetObject();
  SetNone();
  Fetch();
  Print();
  Format();
  BadArgument();
  NoMemory();
  NewException2();
  NewException(m);
}
