#include "torch/script.h"

// see torch/csrc/jit/api/module.h
// torch/csrc/jit/api/*.h

void test() {
  std::string filename = "../foo.pt";
  torch::jit::script::Module module;
  module = torch::jit::load(filename);

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({2, 3}));
  torch::Tensor output = module.forward(inputs).toTensor();
  std::cout << output << "\n";
  torch::Tensor expected = inputs[0].toTensor() + 3;
  assert(output.allclose(expected));

  std::cout << "i: " << module.attr("i") << "\n";
  std::cout << "k: " << module.attr("k") << "\n";
  std::cout << "s: " << module.attr("s") << "\n";
  std::cout << "f: " << module.attr("f") << "\n";
  inputs.push_back(torch::IValue(10));
  output = module.get_method("test2")(inputs).toTensor();
  expected = inputs[0].toTensor() + inputs[1].toInt();
  assert(output.allclose(expected));

  std::cout << output << "\n";

  std::cout << "is training: " << module.is_training() << "\n";
  module.eval();
  std::cout << "is training: " << module.is_training() << "\n";
}

int main() {
  test();
  return 0;
}
