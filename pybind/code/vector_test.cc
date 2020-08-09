#include <sstream>

#include "header.h"

void TestVectorOpaque(py::module& m) {
  py::class_<std::vector<int>>(m, "IntVector");
  m.def("create_int_vec", []() {
    std::vector<int> v{1, 2, 3};
    return v;
  });

  m.def("append_int_vec", [](std::vector<int>* v, int i) { v->push_back(i); });
  m.def("print_int_vec", [](const std::vector<int>& v) {
    std::ostringstream os;
    for (const auto i : v) os << i << " ";
    return os.str();
  });
}
