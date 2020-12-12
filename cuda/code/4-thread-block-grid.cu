
// from THCGeneral.c
// THCudaGetGridSize

void GetGridSize(int32_t *out_num_block_per_column,
                 int32_t *out_num_block_per_row,
                 int32_t *out_num_thread_per_block, size_t size) {

  constexpr int32_t kThreadPerBlock = 256;

  int32_t num_block_per_grid = size / kThreadPerBlock;

  int32_t num_block_per_row = 0;
  int32_t num_block_per_column = 0;

  if (size % kThreadPerBlock)
    ++num_block_per_grid;

  if (num_block_per_grid <= 65535) {
    *out_num_block_per_row = num_block_per_grid;
    *out_num_block_per_column = 1;
  } else if (num_block_per_grid < (65535L * 65535L)) {

    int32_t s = sqrt(num_block_per_grid);
    num_block_per_row = s;
    num_block_per_column = s;
    while (num_block_per_row * num_block_per_column < num_block_per_grid) {
      ++num_block_per_row;
    }
  } else {
    printf("size: %ld is too large!\n", size);
  }

  *out_num_block_per_row = num_block_per_row;
  *out_num_block_per_column = num_block_per_column;
  *out_num_thread_per_block = kThreadPerBlock;
}
