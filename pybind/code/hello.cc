
#include "header.h"

int add(int i, int j) { return i + j; }

extern void TestHandle();
extern void TestObject();
extern void TestAttr();

extern void TestArray(py::module &m);
extern void TestVectorOpaque(py::module &m);
extern void TestBasics(py::module &m);
extern void TestClasses(py::module &m);

PYBIND11_MODULE(hello, m) {
  m.doc() = "pybind11 hello world"; // optional module docstring

  m.def("add", &add, "A function which adds two numbers");

  TestHandle();
  TestObject();
  TestAttr();
  TestArray(m);

  TestVectorOpaque(m);
  TestBasics(m);
  TestClasses(m);
}
