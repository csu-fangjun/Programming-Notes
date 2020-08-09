
#include <iostream>
#include <vector>
#include <string>

__global__ void hello_id(int *p) {

  int id = blockDim.x * blockIdx.x + threadIdx.x;
  p[id] = id;
}

int main() {
  int num_blocks= 5;
  int threads_per_block = 3;
  int n = num_blocks * threads_per_block;

  std::vector<int> v(n);
  int* d;
  cudaError_t ret = cudaMalloc(&d, sizeof(int) * n);
  ret  = cudaMemcpy(d, v.data(), sizeof(int)*n, cudaMemcpyHostToDevice);

  // inside the kernel,
  // gridDim.x == num_blocks
  // gridDim.y == 1
  // gridDim.z == 1

  // blockDim.x == threads_per_block
  // blockDim.y == 1
  // blockDim.z == 1
  hello_id<<<num_blocks, threads_per_block>>>(d);

  // gridDim.x limits the range of blockIdx.x
  // blockDim.x limits the range of threadIdx.x

  ret  = cudaMemcpy(v.data(), d, sizeof(int)*n, cudaMemcpyDeviceToHost);

  std::cout << "host v.data(): " << v.data() << "\n"; // 0x1f09e10
  std::cout << "device d: " << d << "\n"; // 0x7fc2a6c00000

  std::string sep = "";
  for (auto i : v) {
    std::cout << sep << i;
    sep = " ";
  }
  std::cout << "\n";

  ret = cudaFree(d);
}
