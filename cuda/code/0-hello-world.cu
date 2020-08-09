#include <stdio.h>

__global__ void hello_world() {
  printf("hello world\n");
}

int main() {
  hello_world<<<1, 1>>>();

  // the printf function in the kernel prints things
  // in a buffer. We use the following statement to wait
  // for the buffer to be copied from GPU to CPU.
  cudaDeviceReset();

  // Note that there is no need to include header files
  // for cudaDeviceReset
  return 0;
}
