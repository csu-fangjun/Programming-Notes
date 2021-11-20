#include <stdio.h>

// __global__ means it can be called from either host or device
//
// It is always run on device
__global__ void hello_world() { printf("hello world\n"); }

int main() {
  // <<<num_blocks, threads_per_block>>>
  hello_world<<<1, 1>>>();

  // the printf function in the kernel prints things
  // in a buffer. We use the following statement to wait
  // for the buffer to be copied from GPU to CPU.
  cudaDeviceReset();

  // Note that there is no need to include header files
  // for cudaDeviceReset
  return 0;
}
