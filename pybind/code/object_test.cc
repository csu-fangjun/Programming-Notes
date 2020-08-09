
#include <cassert>
#include <iostream>

#include "header.h"

// test include/pybind11/pytypes.h

static void TestObjectImpl() {
  static_assert(std::is_base_of<py::handle, py::object>::value, "");
  PyObject* s = PyUnicode_FromString("hello");
  assert(Py_REFCNT(s) == 1);

  {
      // there are compiler warnings if we assign a PyObject* to py::object
      // py::object o(s, false); // is_borrowed == false, i.e, it steals it
      // assert(Py_REFCNT(s) == 1);
  }

  {
    // it borrows a reference, so it increase the reference count
    py::object o = py::reinterpret_borrow<py::object>(s);
    assert(Py_REFCNT(s) == 2);
    // in the destructor, it decreases the reference count.
  }

  {
    py::object o = py::reinterpret_steal<py::object>(s);
    assert(Py_REFCNT(s) == 1);
    o.inc_ref();
    assert(Py_REFCNT(s) == 2);
    // in the destructor, it decreases the reference count.
  }

  {
    // copy
    py::object o = py::reinterpret_borrow<py::object>(s);
    assert(Py_REFCNT(s) == 2);

    py::object p(o);
    assert(Py_REFCNT(s) == 3);

    // move
    py::object q(std::move(p));
    assert(Py_REFCNT(s) == 3);
    assert(p.ptr() == nullptr);
  }

  {
    // release
    py::object o = py::reinterpret_borrow<py::object>(s);
    assert(Py_REFCNT(s) == 2);

    py::handle h = o.release();
    assert(Py_REFCNT(s) == 2);
    assert(o.ptr() == nullptr);

    // since handle will not decrease the reference count in the constructor,
    // we have to decrease it manually.
    h.dec_ref();
    assert(Py_REFCNT(s) == 1);
  }

  assert(Py_REFCNT(s) == 1);
  Py_XDECREF(s);
}

// error_scope is defined in common.h

void TestErrorScope() {
  PyErr_SetString(PyExc_RuntimeError, "hello");
  assert(PyErr_Occurred() != nullptr);

  {
    py::error_scope scope;  // it calls PyErr_Fetch in the constructor
                            // and PyErr_Restore in the destructor
    assert(PyErr_Occurred() == nullptr);
  }
  assert(PyErr_Occurred() != nullptr);
  PyErr_Clear();
}

// error_string() is defined in cast.h
void TestErrorAlreadySet() {
  PyErr_SetString(PyExc_RuntimeError, "hello (ignore this)");
  assert(PyErr_Occurred() != nullptr);
  std::string s = py::detail::error_string();
  // std::cout << s << "\n"; // RuntimeError: hello (ignore this)
  PyErr_Clear();
}

void TestPyNone() { assert(py::detail::PyNone_Check(Py_None) == 1); }

void TestObject() {
  TestObjectImpl();
  TestErrorScope();
  TestErrorAlreadySet();
  TestPyNone();
}
