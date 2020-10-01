
#include "header.h"

namespace {

struct Foo {};
struct Bar {};

} // namespace

void TestClasses(py::module &m) {
  // note that we can bind the class Foo
  // after the function `foo`!!
  m.def("foo", []() -> Foo { return Foo(); });
  py::class_<Foo>(m, "Foo").def(py::init<>());
}
