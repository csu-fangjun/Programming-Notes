#include <cassert>

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/Tensor.h"
#include "utils/Utils.h"

namespace {

void test_tensor_shape() {
  int ncols = 2;
  int nrows = 3;
  arm_compute::TensorShape shape{ncols, nrows};
  assert(shape.x() == ncols);
  assert(shape.y() == nrows);
  assert(shape[0] == ncols);
  assert(shape[1] == nrows);

  assert(shape.total_size() == ncols * nrows);
  assert(shape.total_size_upper(0) == ncols * nrows);
  assert(shape.total_size_upper(1) == nrows);

  assert(shape.total_size_lower(0) == 1);
  assert(shape.total_size_lower(1) == ncols);
  assert(shape.total_size_lower(2) == ncols * nrows);

  assert(shape.num_dimensions() == 2);
}

void test_tensor_info() {
  arm_compute::TensorInfo info;
  assert(info.data_type() == arm_compute::DataType::UNKNOWN);
  assert(info.format() == arm_compute::Format::UNKNOWN);
  assert(info.tensor_shape().num_dimensions() == 0);
  assert(info.num_dimensions() == 0);
  assert(info.num_channels() == 0);
  assert(info.offset_first_element_in_bytes() == 0);

  assert(arm_compute::num_channels_from_format(arm_compute::Format::F32) == 1);

  int ncols = 2;
  int nrows = 3;
  info = arm_compute::TensorInfo(ncols, nrows, arm_compute::Format::F32);
  assert(info.data_type() == arm_compute::DataType::F32);
  assert(info.num_channels() == 1);
  assert(info.element_size() == sizeof(float));  // 4 * num_channels()
  assert(info.num_dimensions() == 2);

  // total size in bytes
  assert(info.total_size() == ncols * nrows * info.element_size());
  assert(info.tensor_shape().x() == ncols);
  assert(info.tensor_shape().y() == nrows);
}

void test_tensor() {
  arm_compute::TensorInfo info({2, 3}, arm_compute::Format::F32);
  size_t alignment = 0;  // 0 means 64

  arm_compute::Tensor tensor;
  tensor.allocator()->init(info, 0);
  tensor.allocator()->allocate();
  tensor.print(std::cout);
  float* buffer = reinterpret_cast<float*>(tensor.buffer());
  for (int i = 0; i != tensor.info()->tensor_shape().total_size(); ++i) {
    buffer[i] = i;
  }
  tensor.print(std::cout);

  std::vector<float> d = {10, 20, 30, 40, 50, 60};
  tensor.allocator()->import_memory(d.data(), d.size() * sizeof(float));
  tensor.print(std::cout);
  d[0] = 100;
  assert(reinterpret_cast<float*>(tensor.buffer())[0] == d[0]);
  tensor.print(std::cout);
}

void test_neon_sgemm() {
  using namespace arm_compute;
#if 0
  1 0 1  *     1 0 1 0    =   2 1 1 1
  1 1 0        0 2 1 0        1 2 2 0
               1 1 0 1

#endif
  int M = 2;
  int K = 3;
  int N = 4;
  // MxK  * KxN = MxN
  Tensor a{};
  Tensor b{};
  Tensor dst{};
  a.allocator()->init(TensorInfo{TensorShape{K, M}, 1, DataType::F32});
  b.allocator()->init(TensorInfo{TensorShape{N, K}, 1, DataType::F32});
  dst.allocator()->init(TensorInfo{TensorShape{N, M}, 1, DataType::F32});

  std::vector<float> da{1, 0, 1, 1, 1, 0};
  std::vector<float> db{1, 0, 1, 0, 0, 2, 1, 0, 1, 1, 0, 1};
  a.allocator()->import_memory(da.data(), da.size() * sizeof(float));
  b.allocator()->import_memory(db.data(), da.size() * sizeof(float));
  dst.allocator()->allocate();

  float alpha = 1;
  float beta = 0;

  // NEGEMM supports only F16/F32.
  // see
  // https://arm-software.github.io/ComputeLibrary/v18.11/classarm__compute_1_1_n_e_g_e_m_m_matrix_multiply_kernel.xhtml#a4ee07709711414457834cc5d1c2c6cdb
  NEGEMM sgemm{};
  // https://arm-software.github.io/ComputeLibrary/v20.02.1/classarm__compute_1_1_n_e_g_e_m_m.xhtml#a385241dcc5062af6ecac8bdafe01bb2a
  // dst = alpha * a * b + beta * c
  // since c is nullptr, so dst = alpha * a * b
  sgemm.configure(&a, &b, nullptr, &dst, alpha, beta);
  sgemm.run();
  dst.print(std::cout);

  da[0] = 10;
  sgemm.run();
  dst.print(std::cout);
}

void test_neon_sgemm_s8() {
  using namespace arm_compute;
  std::cout << "for sgemm s8\n";
#if 0
  1 0 1  *     1 0 1 0    =   2 1 1 1
  1 1 0        0 2 1 0        1 2 2 0
               1 1 0 1

#endif
  int M = 2;
  int K = 3;
  int N = 4;
  // MxK  * KxN = MxN
  Tensor a{};
  Tensor b{};
  Tensor dst{};
  a.allocator()->init(TensorInfo{TensorShape{K, M}, 1, DataType::S8});
  b.allocator()->init(TensorInfo{TensorShape{N, K}, 1, DataType::S8});
  dst.allocator()->init(TensorInfo{TensorShape{N, M}, 1, DataType::S32});

  std::vector<int8_t> da{1, 0, 1, 1, 1, 0};
  std::vector<int8_t> db{1, 0, 1, 0, 0, 2, 1, 0, 1, 1, 0, 1};
  a.allocator()->import_memory(da.data(), da.size());
  b.allocator()->import_memory(db.data(), db.size());
  dst.allocator()->allocate();

  NEGEMMLowpMatrixMultiplyCore sgemm{};
  // use default value for GEMMInfo
  sgemm.configure(&a, &b, nullptr, &dst);
  sgemm.run();
  dst.print(std::cout);

  da[0] = 10;
  sgemm.run();
  dst.print(std::cout);
}

void test() {
  test_tensor_shape();
  test_tensor_info();
  test_tensor();
  test_neon_sgemm();
  test_neon_sgemm_s8();
  return;
}
}  // namespace

int main() {
  test();
  return 0;
}
