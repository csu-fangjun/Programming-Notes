#include <stdint.h>
#include <stdio.h>      // for select
#include <sys/select.h> // for select
#include <sys/time.h>

// man select
//
// If we specify a timeout, e.g., it is not a null pointer,
//  - If it returns 0, it means it timeouts and there are no fds that are ready
//  - Otherwise, it returns the number fds that are ready to read/write
//  - Pass a timeout with all zeor will return immediately
//
// If we don't specify a timeout, e.g., pass a null pointer, it waits
// indefinitely
//
// It returns -1 on failure

int main() {
  fd_set readfds, write_fds;
  FD_ZERO(&readfds);
  FD_ZERO(&write_fds);
  FD_SET(0, &readfds);
  struct timeval time;
  time.tv_sec = 3; // wait for 3 seconds. If set to 0, it returns immediately
  time.tv_usec = 0;
  // the max fd is 0, so we use 1 here
  // the max valid fd is FD_SETSIZE - 1
  int32_t ret = select(1, &readfds, NULL, NULL, &time);
  printf("ret: %d\n", ret);
  if (FD_ISSET(0, &readfds)) {
    printf("fd 0 is ready for read\n");
    char buf[100];
    char *line = fgets(buf, 100, stdin);
    printf("read %s\n", line);

    // Note: time out is changed in place, though it is not portable.
    // Some implementations don't change it
    printf("remaining time: %ld s, %ld us\n", time.tv_sec, time.tv_usec);
  } else {
    printf("fd 0 is NOT ready for read\n");
  }

  // NOTE: we have to read all inputs from the fd 0, otherwise, the following
  // select returns immediately since fd 0 is ready to read

  FD_ZERO(&readfds);
  FD_SET(0, &readfds);
  ret = select(1, &readfds, NULL, NULL, NULL); // wait indefinitely
  printf("ret: %d\n", ret);
  if (FD_ISSET(0, &readfds)) {
    printf("fd 0 is ready for read\n");
    char buf[100];
    char *line = fgets(buf, 100, stdin);
    printf("read %s\n", line);
  }
}
