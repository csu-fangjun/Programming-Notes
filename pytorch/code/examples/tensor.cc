#include <iostream>

#include "c10/core/DispatchKeySet.h"
#include "torch/torch.h"

static void test_tensor_size() {
  static_assert(sizeof(at::Tensor) == 8,
                "it contains only an intrusive pointer");
  at::Tensor t = at::tensor(123);

  // it overloads operator== in c10/core/ScalarType.h
  assert(t.dtype() == torch::kInt);

  assert(t.numel() == 1);
  assert(t.dim() == 1);
  assert(t.sizes() == (std::vector<int64_t>{1}));
  assert(t.item<int>() == 123);
  std::cout << t.sizes() << "\n";
}

static void test_tensor_impl() {
  // for torch < 1.8.0
  // static_assert(sizeof(c10::TensorImpl) == 248,
  //               "it should contain 31 * sizeof(int64_t)");
  //
  // Storagge storage_;
  // int64_t storage_offset_ = 0;  // number of elements, NOT number of bytes
  // int64_t numel_ = 1;
  // caffe2::TypeMeta data_type_;
  // c10::optional<c10::Device> device_opt_;
  // DispatchKeySet key_set_;
  // bool is_contiguous_ = true;
  //
  // SmallVector<int64_t, 5> sizes_;
  // SmallVector<int64_t, 5> strides_;
  //
  // bool is_channels_last_ = false;
  // bool is_channels_last_contiguous = false_;
  // bool is_channels_last_3d_ = false;
  // bool is_non_overlapping_and_dense_ = false;
  // bool is_wrapped_number_ = false;
  // bool allow_tensor_metadata_change_ = false;
  // bool reserved_ = false;
}

template <typename T> void foo() { std::cout << __PRETTY_FUNCTION__ << "\n"; }

static void test_dispatch_key_set() {
  static_assert(sizeof(c10::DispatchKeySet) == 8,
                "It contains only an uint64_t member!");
  c10::DispatchKeySet s({c10::DispatchKey::CPU, c10::DispatchKey::CUDA});
  assert(s.raw_repr() == 3);
  std::cout << s << "\n"; // DispatchKeySet(CUDA, CPU)

  foo<int>();
  foo<int32_t>();
  foo<torch::Tensor>();
}

static void test_type_meta() {
  caffe2::TypeMeta t1 = caffe2::TypeMeta::Make<int>();
  assert(t1.name() == std::string("int"));
}

static void test_index() {
  torch::Tensor a = torch::arange(8).reshape({2, 4});
  torch::Tensor b = a.index({"...", -1});
  std::cout << a;
  std::cout << b;
  torch::Tensor c = torch::arange(2);
  b.copy_(c);
  std::cout << "\nafter copying\n";
  std::cout << a;
  std::cout << b;
}

void test_tensor() {
  test_type_meta();
  test_dispatch_key_set();
  test_tensor_size();
  test_tensor_impl();
  test_index();
}
