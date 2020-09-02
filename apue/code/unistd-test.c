#include <stdio.h>
#include <unistd.h>

// refer to
// https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/unistd.h.html
//
//
// unistd.h is not a C standard; it is defined by posix.

void test() {
#if _POSIX_VERSION
  printf("_POSIX_VERSION: %ld\n", _POSIX_VERSION); // 200809
#endif

#if _POSIX2_VERSION
  printf("_POSIX2_VERSION: %ld\n", _POSIX2_VERSION); // 200809
#endif

#if _XOPEN_VERSION
  printf("_XOPEN_VERSION: %d\n", _XOPEN_VERSION); // 700
#endif

  printf("STDIN_FILENO: %d\n", STDIN_FILENO);   // 0
  printf("STDOUT_FILENO: %d\n", STDOUT_FILENO); // 1
  printf("STDERR_FILENO: %d\n", STDERR_FILENO); // 2
}

int main() {
  test();
  return 0;
}
