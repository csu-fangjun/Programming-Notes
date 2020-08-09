#include <cassert>
#include <iostream>

#include "header.h"

static int g_a = 3;

static void TestPointers(py::module& m) {
  m.def("inc", [](int* p) { std::cout << "input p is: " << p << "\n"; });
  m.def("inc", [](int* p) { std::cout << "input p2 is: " << p << "\n"; });
}

class Foo {};

static void TestClasses(py::module& m) {
  py::class_<Foo>(m, "Foo2");
  m.def("Hi", [](const Foo& f) {});
}

void TestBasics(py::module& m) {
  TestPointers(m);
  TestClasses(m);
}
