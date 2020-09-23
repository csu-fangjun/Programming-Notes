#include <iostream>

#include "c10/cuda/CUDAFunctions.h"
#include "torch/torch.h"

//
// at::detail::getCUDAHooks, aten/src/ATen/detail/CUDAHooksInterface
// at::cuda::globalContext().lazyInitCUDA()

static void test_device_count() {
  std::cout << "num gpus: " << c10::cuda::device_count() << "\n";
  std::cout << "current device: " << c10::cuda::current_device() << "\n";
}

static void test() {
  at::Tensor a = at::ones({2, 3}, torch::kFloat);
  at::Tensor b = at::ones({2, 3}, torch::kFloat);
  at::Tensor c = a + b;
}

void test_cuda() {
  //
  // test_device_count();
  test();
}
