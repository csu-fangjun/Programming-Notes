#include <cassert>

#include "header.h"

// test include/pybind11/pytypes.h
//
// there is no reference counting for py::handle!

static void TestImpl() {
  static_assert(py::detail::is_pyobject<py::handle>::value, "");

  PyObject* o = PyUnicode_FromString("hello");
  assert(Py_REFCNT(o) == 1);

  {
    py::handle h(o);            // it steals the reference
    assert(Py_REFCNT(o) == 1);  // still 1
    assert(h.ptr() == o);

    h.inc_ref();
    assert(Py_REFCNT(o) == 2);  // now it is 2!

    h.dec_ref();
    assert(Py_REFCNT(o) == 1);  // now it is 1!
    assert(h);                  // ptr is not null, so h is true
  }
  {
    py::handle h;
    assert(h.ptr() == nullptr);
    assert(!h);  // ptr is null, so h is false
  }

  Py_XDECREF(o);
}

void TestHandle() { TestImpl(); }
