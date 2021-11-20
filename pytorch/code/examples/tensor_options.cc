
#include "torch/torch.h"

// see c10/core/TensorOptions.h
//
// device, dtype (caffe2::TypeMeta), requires_grad
// layout, memory_format

static void test() {
#if 0
  torch::TensorOptions opt = torch::dtype(torch::kInt);
  // note: opt.dtype() is of type caffe2::TypeMeta
  assert(opt.dtype() == caffe2::TypeMeta::fromScalarType(torch::kInt));

  // It is recommended to use torch::dtype.
  // Don't call such constructors directly
  torch::TensorOptions opt2 = torch::TensorOptions(torch::kInt);
  torch::TensorOptions opt3 = torch::TensorOptions().dtype(torch::kInt);

  torch::TensorOptions opt4 = opt3.device(torch::kCPU);

  // recommended
  torch::TensorOptions opt5 = torch::device(torch::kCPU).dtype(torch::kInt);
  assert(opt5.device() == torch::Device(torch::kCPU));

  torch::TensorOptions opt6 = torch::requires_grad(true);
  opt6 = opt5.requires_grad(false);
#endif
}

void test_tensor_options() { test(); }
