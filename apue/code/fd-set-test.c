#include <assert.h>
#include <stdio.h>
#include <sys/select.h>

// see

int main() {
  printf("FD_SETSIZE: %d\n", FD_SETSIZE); // 1024
  // Note the maximum valid file description is 1023, i.e., FD_SETSIZE - 1
  fd_set fds;
  FD_ZERO(&fds); // clear all bits
  assert(FD_ISSET(10, &fds) == 0);

  FD_SET(10, &fds); // set fd 10
  assert(FD_ISSET(10, &fds) == 1);

  FD_CLR(10, &fds); // clear fd 10
  assert(FD_ISSET(10, &fds) == 0);

  return 0;
}
