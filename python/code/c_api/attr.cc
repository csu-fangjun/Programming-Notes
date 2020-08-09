#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cassert>

static void AddIntAttribute(PyObject* m) {
  {
    // case 1: Use PyModule_AddIntConstant
    const char* name = "i1";
    long value = 10;
    assert(PyModule_Check(m) == 1);
    PyModule_AddIntConstant(m, name, value);
  }
  {
    // case 2: use PyModule_AddObject
    const char* name = "i2";
    long value = -2000;
    auto o = PyLong_FromLong(value);
    assert(Py_REFCNT(o) == 1);
    PyModule_AddObject(m, name, o);  // it **steals** the reference!
    assert(Py_REFCNT(o) == 1);       // it is still 1!
  }
  {
    // case 3: use PyObject_SetAttrString
    const char* name = "i3";
    long value = 10000;
    auto o = PyLong_FromLong(value);
    assert(Py_REFCNT(o) == 1);
    PyObject_SetAttrString(m, name, o);
    assert(Py_REFCNT(o) == 2);  // ref cnt of o is increased by 1 !
    Py_XDECREF(o);
    assert(Py_REFCNT(o) == 1);
  }
  {
    // case 4: use PyObject_SetAttr
    const char* name = "i4";
    auto key = PyUnicode_FromString(name);
    assert(Py_REFCNT(key) == 1);
    assert(PyUnicode_Check(key) == 1);

    long value = 100000;
    auto o = PyLong_FromLong(value);
    assert(Py_REFCNT(o) == 1);

    PyObject_SetAttr(m, key, o);

    // assert(Py_REFCNT(key) == 1);
    assert(Py_REFCNT(o) == 2);  // note ref cnt of o is increased!

    Py_XDECREF(o);
    Py_XDECREF(key);
  }

  // lookup
  {
    const char* name = "i1";
    assert(PyObject_HasAttrString(m, name) == 1);
    assert(PyObject_HasAttrString(m, "i2") == 1);
    assert(PyObject_HasAttrString(m, "i3") == 1);
    assert(PyObject_HasAttrString(m, "i4") == 1);

    assert(PyObject_HasAttrString(m, "i100") == 0);

    auto key = PyUnicode_FromString(name);
    assert(PyObject_HasAttr(m, key) == 1);

    Py_XDECREF(key);
  }

  // GetAttr
  {
    auto o = PyObject_GetAttrString(m, "i4");  // we have to release o
    assert(PyLong_Check(o) == 1);
    assert(Py_REFCNT(o) == 2);

    assert(PyLong_AsLong(o) == 100000);

    Py_XDECREF(o);  // we have to release it!
  }
  {
    auto name = PyUnicode_FromString("i4");
    auto o = PyObject_GetAttr(m, name);
    assert(PyLong_Check(o) == 1);
    assert(Py_REFCNT(o) == 2);

    assert(PyLong_AsLong(o) == 100000);
    Py_XDECREF(o);
  }
}

static void AddStringAttribute(PyObject* m) {
  {
    // case 1: Use PyModule_AddStringConstant
    const char* name = "s1";
    const char* value = "S1";
    PyModule_AddStringConstant(m, name, value);

    assert(PyObject_HasAttrString(m, name) == 1);

    // now get the attr
    PyObject* key = PyUnicode_FromString(name);
    assert(Py_REFCNT(key) == 1);
    assert(PyObject_HasAttr(m, key) == 1);
    PyObject* v = PyObject_GetAttr(m, key);
    assert(Py_REFCNT(v) == 2);
    assert(PyUnicode_Check(v) == 1);
    Py_XDECREF(v);
    Py_XDECREF(key);  // we have to release the key!

    assert(PyObject_HasAttrString(m, "s1") == 1);

    v = PyObject_GetAttrString(m, name);
    assert(Py_REFCNT(v) == 2);
    assert(PyUnicode_Check(v) == 1);
    assert(PyUnicode_CheckExact(v) == 1);

    // No need to free the returned pointer!
    const char* dv = PyUnicode_AsUTF8(v);

    assert(strcmp(dv, "S1") == 0);
    Py_XDECREF(v);  // we have to free the returned v!
  }
}

static void TestObjectAttr(PyObject* m) {
  PyObject* key = PyUnicode_FromString("key");
  PyObject* value = PyUnicode_FromString("value");

  assert(Py_REFCNT(key) == 1);
  assert(Py_REFCNT(value) == 1);

  // Note that we can use PyModule_AddObject to add an attr
  // and later we use     PyObject_GetAttr to get the attr.

  PyModule_AddObject(m, "key", value);  // it steals the reference
  assert(Py_REFCNT(value) == 1);

  PyObject* v = PyObject_GetAttr(m, key);  // return a new reference
  assert(Py_REFCNT(v) == 2);
  const char* dv = PyUnicode_AsUTF8(v);
  assert(strcmp(dv, "value") == 0);
  Py_XDECREF(v);

  Py_XDECREF(key);
  Py_XDECREF(value);
}

void TestAttr(PyObject* m) {
  assert(PyModule_Check(m) == 1);
  AddIntAttribute(m);
  AddStringAttribute(m);
  TestObjectAttr(m);
}
