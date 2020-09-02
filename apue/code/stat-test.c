#include <assert.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int main() {
  const char *filename = "./stat-test.c";
  struct stat buf;
  int r = stat(filename, &buf);
  assert(r == 0);
  printf("st_size: %ld (file size in bytes)\n", buf.st_size);
  printf("st_blksize: %ld\n", buf.st_blksize);
  printf("st_blocks: %ld\n", buf.st_blocks);
}
