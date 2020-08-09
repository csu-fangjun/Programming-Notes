#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include <cassert>
#include <cstring>
#include <iostream>

// create a module `PyModule`.
// The only thing we need to create a module is its
// name, which is passed by `PyModuleDef`.
static PyObject* CreateModule() {
  // it is never freed
  auto module_def = new PyModuleDef;
  std::memset(module_def, 0, sizeof(PyModuleDef));

  assert(Py_REFCNT(module_def) == 0);

  // only `m_name` is required
  module_def->m_name = "hello";  // it will set __name__ of the module

  // optionally, we can set the `m_doc`
  module_def->m_doc = "doc of hello";

  auto m = PyModule_Create(module_def);

  // it sets the reference count of `module_def` to 1 inside `PyModule_Create`

  assert(Py_REFCNT(module_def) == 1);
  assert(Py_REFCNT(m) == 1);

  // it is a module
  assert(PyModule_Check(m) == 1);

  // it is a module and not a subclass of module
  assert(PyModule_CheckExact(m) == 1);

  return m;
}

extern void TestAttr(PyObject* m);
extern void TestException(PyObject* m);
extern void TestMethods(PyObject* m);
extern void TestMacros(PyObject* m);
extern void TestList(PyObject* m);
extern void TestTuple(PyObject* m);
extern void TestDict(PyObject* m);
extern void TestNewType(PyObject* m);
extern void TestNewTypeWithMethods(PyObject* m);

// it returns a PyObject*
// The name has to be PyInit_<module_name>;
// In python, we use `import module_name`
PyMODINIT_FUNC PyInit_hello() {
  auto m = CreateModule();

  TestAttr(m);
  TestException(m);
  TestMethods(m);
  TestMacros(m);
  TestList(m);
  TestTuple(m);
  TestDict(m);
  TestNewType(m);
  TestNewTypeWithMethods(m);

  return m;
}
