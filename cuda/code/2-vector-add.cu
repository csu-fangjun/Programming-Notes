
#include<vector>
#include<cassert>

void cpu_vec_add(const int* a, const int* b, int* c, int n) {
  for (int i = 0; i != n; ++i) {
    c[i] = a[i] + b[i];
  }
}

__global__ void gpu_vec_add(const int* a, const int* b, int* c, int n){
  // we use a 1-d grid, 1-d block
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n)
  {
    c[id] = a[id] + b[id];
  }
}

int main() {
  int n = 10;
  int threads_per_block = 3;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;

  std::vector<int> a(n);
  std::vector<int> b(n);
  std::vector<int> c(n);
  std::vector<int> ground_truth(n);
  for (int i = 0; i != n; ++i) {
    a[i] = i;
    b[i] = 2*i;
  }

  cpu_vec_add(a.data(), b.data(), ground_truth.data(), n);

  int *da;
  int *db;
  int *dc;
  cudaMalloc(&da, sizeof(int) * n);
  cudaMalloc(&db, sizeof(int) * n);
  cudaMalloc(&dc, sizeof(int) * n);


  cudaMemcpy(da, a.data(), sizeof(int)*n, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b.data(), sizeof(int)*n, cudaMemcpyHostToDevice);

  gpu_vec_add<<<num_blocks, threads_per_block>>>(da, db, dc, n);

  cudaMemcpy(c.data(), dc, sizeof(int)*n, cudaMemcpyDeviceToHost);

  cudaFree(da);
  cudaFree(db);

  for(int i = 0; i !=n; ++i) {
    assert(c[i] == ground_truth[i]);
  }
  cudaProfilerStop();
  return 0;
}
