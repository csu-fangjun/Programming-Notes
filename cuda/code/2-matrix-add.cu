#include <cassert>
#include <iostream>
#include <vector>

void cpu_matrix_add(const int *a, const int *b, int *c, int num_rows,
                    int num_cols) {
  int n = num_rows * num_cols;
  for (int i = 0; i != n; ++i) {
    c[i] = a[i] + b[i];
  }
}

__global__ void gpu_matrix_add(const int *a, const int *b, int *c, int num_rows,
                               int num_cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < num_rows && col < num_cols) {
    int index = row * num_cols + col;
    c[index] = a[index] + b[index];
  }
}

int main() {
  constexpr int kNumRows = 500;
  constexpr int kNumCols = 1000; // TODO: segfault if using 5000
  int a[kNumRows][kNumCols];
  int b[kNumRows][kNumCols];
  int gpu[kNumRows][kNumCols];
  int cpu[kNumRows][kNumCols];

  for (int i = 0; i != kNumRows * kNumCols; ++i) {
    a[0][i] = i;
    b[0][i] = i * 2;
  }
  cpu_matrix_add(a[0], b[0], cpu[0], kNumRows, kNumCols);

  int *da;
  int *db;
  int *dc;

  // ignore error check
  cudaMalloc(&da, sizeof(int) * kNumRows * kNumCols);
  cudaMalloc(&db, sizeof(int) * kNumRows * kNumCols);
  cudaMalloc(&dc, sizeof(int) * kNumRows * kNumCols);

  cudaMemcpy(da, a, sizeof(int) * kNumRows * kNumCols, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, sizeof(int) * kNumRows * kNumCols, cudaMemcpyHostToDevice);

  int threads_per_block_x = 16;
  int threads_per_block_y = 8;
  dim3 threads_per_block(threads_per_block_x, threads_per_block_y);
  assert(threads_per_block.x == threads_per_block_x);
  assert(threads_per_block.y == threads_per_block_y);

  dim3 num_blocks;
  num_blocks.x = (kNumCols + threads_per_block.x - 1) / threads_per_block.x;
  num_blocks.y = (kNumRows + threads_per_block.y - 1) / threads_per_block.y;
  // TODO: handle the case when num_blocks.x and num_blocks.y exceed the limit
  // of the GPU card
  gpu_matrix_add<<<num_blocks, threads_per_block>>>(da, db, dc, kNumRows,
                                                    kNumCols);

  cudaMemcpy(gpu[0], dc, sizeof(int) * kNumRows * kNumCols,
             cudaMemcpyDeviceToHost);

  for (int i = 0; i != kNumRows * kNumCols; ++i) {
    assert(cpu[0][i] == gpu[0][i]);
  }
}
