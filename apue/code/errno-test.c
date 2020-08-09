#include <errno.h>   // for errno
#include <stdio.h>   // for perror
#include <string.h>  // for strerror

void test_perror() {
  errno = EPERM;
  perror("test_perror");  // test_error: Operation not permitted
  errno = 0;
}

void test_strerror() {
  // we cannot modify the string returned by strerror
  // it may be modified by the subsequent strerror.
  char* s = strerror(EPERM);  // Operation not permited
  printf("%s\n", s);
}
int main() {
  test_perror();
  test_strerror();
  return 0;
}
#if 0
test_perror: Operation not permitted
Operation not permitted
#endif
