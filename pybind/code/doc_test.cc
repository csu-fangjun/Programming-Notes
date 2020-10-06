#include "header.h"

/*
todo:
https://github.com/intel-isl/Open3D/pull/864/files
Pybind docs parser and Google-style docstring generator


 */

struct FooDoc {
  FooDoc() = default;
  FooDoc(int f) : f(f) {}
  int f;
};
FooDoc add(int a, const FooDoc *foo = nullptr) { return FooDoc(); }

static void TestFooDoc(py::module &m) {
  py::class_<FooDoc> pyclass(m, "FooDoc");
  pyclass.def(py::init<int>(), py::arg("f"),
              R"(
      Constructor of :class:`FooDoc`.
      )");

  py::object o = m.attr("FooDoc");
  py::object a = o.attr("__init__");
  // a.doc() = strdup("__init__(self, f)");

  std::cout << (py::object)a.doc();
}

void TestDoc(py::module &m) {
  // m.def("doc_add", &add);
  TestFooDoc(m);
}
