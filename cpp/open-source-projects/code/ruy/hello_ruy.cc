// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)
//
// Refer to
//  https://github.com/google/ruy/blob/600d1ecbc15a500916a3efed77db27d5e5d55cb5/ruy/example.cc#L16

#include <iostream>

#include "ruy/ruy.h"
#include "ruy/ruy_advanced.h"  // for prepacked matmul

namespace {

void test1(ruy::Context* context) {
  //
  // 1 2 * 5 7  =  5+12   7+16    = 17  23
  // 3 4   6 8     15+24  15+32     39  47
  std::vector<float> a = {1, 2, 3, 4};
  std::vector<float> b = {5, 6, 7, 8};
  std::vector<float> c(4, 0);

  ruy::Matrix<float> lhs;  // left hand side
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kRowMajor, &lhs.layout);
  lhs.data = a.data();

  ruy::Matrix<float> rhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &rhs.layout);
  rhs.data = b.data();

  ruy::Matrix<float> dst;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &dst.layout);
  dst.data = c.data();

  ruy::BasicSpec<float, float> spec;
  ruy::Mul<ruy::kAllPaths>(lhs, rhs, spec, context, &dst);
  for (auto i : c) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  // 17 39 23 53
  // Note that the result is in column major

  // now with bias
  std::vector<float> bias = {1, -1};
  spec.bias = bias.data();
  ruy::Mul<ruy::kAllPaths>(lhs, rhs, spec, context, &dst);
  for (auto i : c) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  // 18 38 24 52

  // now with clamp
  spec.clamp_min = 20;
  spec.clamp_max = 40;
  ruy::Mul<ruy::kAllPaths>(lhs, rhs, spec, context, &dst);
  for (auto i : c) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  // 20 38 24 40
  //  every thing less than clamp_min is set to clamp_min
  //  every thing greater than clamp_max is set to clamp_max

  // now clamp without bias
  spec.bias = nullptr;
  spec.clamp_min = 16;
  spec.clamp_max = 37;

  ruy::Mul<ruy::kAllPaths>(lhs, rhs, spec, context, &dst);
  for (auto i : c) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  // 17 37 23 37
}

class SimpleAllocator {
 public:
  void* Allocate(size_t num_bytes) {
    std::cout << "allocating " << num_bytes << "\n";
    char* p = new char[num_bytes];
    buffers_.emplace_back(p);
    return p;
  }

 private:
  std::vector<std::unique_ptr<char[]>> buffers_;
};

// now with prepacked matrix
void test2(ruy::Context* context) {
  std::cout << "==========test2==========\n";
  //
  // 1 2 * 5 7  =  5+12   7+16    = 17  23
  // 3 4   6 8     15+24  15+32     39  47
  std::vector<float> a = {1, 2, 3, 4};
  std::vector<float> b = {5, 6, 7, 8};
  std::vector<float> c(4, 0);

  ruy::Matrix<float> lhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kRowMajor, &lhs.layout);

  ruy::Matrix<float> rhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &rhs.layout);

  ruy::Matrix<float> dst;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &dst.layout);

  SimpleAllocator allocator;

  auto allocate_func = [&allocator](size_t num_bytes) -> void* {
    return allocator.Allocate(num_bytes);
  };

  ruy::BasicSpec<float, float> spec;

  ruy::PrepackedMatrix prepacked_rhs;

  rhs.data = b.data();
  ruy::PrePackForMul<ruy::kAllPaths>(lhs, rhs, spec, context, &dst, nullptr,
                                     &prepacked_rhs, allocate_func);

  // data is copied to prepacked _rhs, we do not need b any more.
  rhs.data = nullptr;

  lhs.data = a.data();
  dst.data = c.data();

  ruy::MulWithPrepacked<ruy::kAllPaths>(lhs, rhs, spec, context, &dst, nullptr,
                                        &prepacked_rhs);

  for (auto i : c) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  // 17 39 23 53

  // now set b to zero
  b[0] = b[1] = b[2] = b[3] = 0;

  ruy::MulWithPrepacked<ruy::kAllPaths>(lhs, rhs, spec, context, &dst, nullptr,
                                        &prepacked_rhs);
  for (auto i : c) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  // 17 39 23 53
  //
  // Note that c is not changed!
}

}  // namespace

int main() {
  ruy::Context context;
  test1(&context);
  test2(&context);
  return 0;
}
