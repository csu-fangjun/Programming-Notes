#include <stdio.h>

// WARNING: gets is deprecated as it does not check the buffer length
// the worm attack was caused by gets!

// Note that `gets` discards the new line
void test_gets() {
  char buf[10];
  gets(buf);
  printf("read: %shello\n", buf);
}

// the newline is not discarded!
// it is always ended with \0, so it may replace
// the last \n with \0
void test_fgets() {
  char buf[3];
  // it reads at most sizeof(buf) - 1 bytes until it encounters EOF or \n
  // \n is read into the buf.
  // if the input is 12\n, then it reads only 12 and leave \n in the input
  // buffer
  fgets(buf, sizeof(buf), stdin);
  printf("fgets read: %shello\n", buf);
}

int main() {
  /** test_gets(); */
  test_fgets();
  return 0;
}
