#include <torch/extension.h>

torch::Tensor sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid", &sigmoid, "sigmoid test");
}
