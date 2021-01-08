#include <iostream>
#include <vector>

__global__ void saxpy(int32_t n, float alpha, const float *x, float *y) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= n)
    return;

  y[tid] += alpha * x[tid];
}

int main() {
  std::vector<float> x{1, 2, 3};
  float alpha = 2;
  std::vector<float> y{10, 20, 30};

  float *d_x;
  cudaMalloc(&d_x, x.size() * sizeof(float));
  cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

  float *d_y;
  cudaMalloc(&d_y, x.size() * sizeof(float));
  cudaMemcpy(d_y, y.data(), y.size() * sizeof(float), cudaMemcpyHostToDevice);

  int32_t threads_per_block = 256;
  int32_t num_blokcs = (x.size() + threads_per_block - 1) / threads_per_block;
  saxpy<<<num_blokcs, threads_per_block>>>(x.size(), alpha, d_x, d_y);
  cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost);

  for (auto i : y)
    std::cout << i << "\n";
}
