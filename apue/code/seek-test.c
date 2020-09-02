#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

int main() {
  const char *filename = "./a.txt";
  FILE *fp = fopen(filename, "w+");
  if (!fp) {
    printf("failed to open: %s\n", filename);
    return -1;
  }

  char b[] = "12345";
  fwrite(b, 1, 5, fp);
  rewind(fp);

  char a;
  fread(&a, 1, 1, fp);
  printf("a: %c\n", a); // 1

  int fd = fileno(fp);
  printf("fd: %d\n", fd); // 3

  int off = lseek(fd, 0, SEEK_CUR);
  printf("lseek off: %d\n", off); // 5

  off = ftell(fp);
  printf("ftell off: %d\n", off); // 1

  /*
   Note the different results between lseek and ftell.

   fread keeps a buffer internally and it uses `read` to read a BUFSIZE (usually
   4096 bytes), which affects the `lseek`.

   `lseek` is for `read` and is a system call.
   `ftell` is for libc and is a library call.
   */

  fclose(fp);
  remove(filename);

  return 0;
}
