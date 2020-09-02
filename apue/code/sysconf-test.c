#include <stdio.h>
#include <unistd.h>

int main() {
  long n = sysconf(_SC_ATEXIT_MAX);
  printf("ATEXIT_MAX: %ld, %#lx\n", n, n);
  return 0;
}
