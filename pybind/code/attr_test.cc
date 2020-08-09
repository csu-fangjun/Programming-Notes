#include <cassert>
#include <iostream>

#include "header.h"

// test include/pybind11/pytypes.h

static void TestAttrImpl() {
  PyObject* s = PyUnicode_FromString("foo");

  py::handle h(s);
  assert(py::hasattr(h, "__doc__") == 1);
  std::string doc_str = h.attr("__doc__").cast<std::string>();
}

void TestAttr() { TestAttrImpl(); }
