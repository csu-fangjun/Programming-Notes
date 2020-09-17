#include <cassert>
#include <iostream>
#include <vector>

#include <torch/torch.h>

// refer to https://pytorch.org/cppdocs/notes/tensor_basics.html#cpu-accessors
static void test_accessor() {
  torch::Tensor t = torch::ones({3, 3}, torch::kFloat32);
  // assert its data type is float and ndim is 2
  auto accessor = t.accessor<float, 2>();

  float trace = 0;
  for (int i = 0; i < accessor.size(0); ++i) {
    trace += accessor[i][i];
  }

  assert(trace == 3);
}

// refer to
// https://pytorch.org/cppdocs/notes/tensor_basics.html#using-externally-created-data
static void test_external() {
  float d[] = {0, 1, 2, 3, 4, 5};
  torch::Tensor t = torch::from_blob(d, {2, 3});
}

static void test_scalar() { torch::Scalar t = 1.0; }

// refer to
// https://pytorch.org/cppdocs/notes/tensor_creation.html#specifying-a-size
static void test_sizes() {
  {
    // 1-d tensor
    torch::Tensor t = torch::ones(2);
    assert(t.size(0) == 2);
    assert(t.sizes()[0] == 2);
    assert(t.sizes() == (std::vector<int64_t>{2}));
  }

  {
    // 2-d tensor
    torch::Tensor t = torch::zeros({2, 3});
    assert(t.size(0) == 2);
    assert(t.size(1) == 3);

    assert(t.sizes()[0] == 2);
    assert(t.sizes()[1] == 3);

    assert(t.sizes() == (std::vector<int64_t>{2, 3}));
  }
}

static void test_tensor_options() {
  torch::TensorOptions options =
      torch::TensorOptions()
          .dtype(torch::kFloat32)  // default: kFlaot32
          .layout(torch::kStrided) // default:kStrided
          .device(torch::kCUDA, 0) // default: kCPU
          .requires_grad(true);    // default: false

  auto t = torch::empty({2, 3}, options);
  assert(t.dtype() == torch::kFloat32);
  assert(t.layout() == torch::kStrided);
  assert(t.device().type() == torch::kCUDA);
  assert(t.device().index() == 0);
  assert(t.requires_grad() == true);

  t = torch::empty(2, torch::dtype(torch::kInt32)
                          .layout(torch::kStrided)
                          .device(torch::kCPU)
                          .requires_grad(false));
  assert(t.dtype() == torch::kInt32);
  assert(t.layout() == torch::kStrided);
  assert(t.device().type() == torch::kCPU);
  assert(t.requires_grad() == false);
}

static void test_ref() {
  auto t = torch::zeros(3);
  assert(t.use_count() == 1);

  {
    auto t2 = t;
    assert(t.use_count() == 2);
    assert(t2.use_count() == 2);
  }
  assert(t.use_count() == 1);
}

void test_hello() {
  test_accessor();
  test_external();
  test_scalar();
  test_sizes();
  test_tensor_options();
  test_ref();

  // std::cout << t.device() << "\n";
  // std::cout << t << "\n";
  // t = t.to(torch::kCUDA);
  //
  // std::cout << t << "\n";
}
