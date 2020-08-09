#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*
Test useful macros listed in https://docs.python.org/3.9/c-api/intro.html

They are defined in Include/pymacro.h
*/

static void TestUsefulMacro() {
  assert(Py_ABS(10) == 10);
  assert(Py_ABS(-10) == 10);

  assert(Py_MIN(1, 10) == 1);
  assert(Py_MAX(1, 10) == 10);

  assert(strcmp(Py_STRINGIFY(10), "10") == 0);

  int Py_UNUSED(b);
  int _unused_c __attribute__((unused));
  int d;
}

void TestMacros(PyObject* m) { TestUsefulMacro(); }
