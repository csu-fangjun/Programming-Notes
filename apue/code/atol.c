#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_atoi() {
  int i = atoi("12");
  assert(i == 12);

  i = atoi("abc");
  assert(i == 0);
}

static void test_strtol() {
  char* endptr;
  const char* p = "123";
  errno = 0;
  long int i = strtol(p, &endptr, 10);
  assert(i == 123);
  assert(endptr == p + 3);
  assert(*endptr == '\0');
  assert(errno == 0);

  i = strtol(p, &endptr, 16);
  assert(i == 0x123);
  assert(*endptr == '\0');
  assert(errno == 0);

  p = "   +123";
  i = strtol(p, &endptr, 10);
  assert(i == 123);
  assert(endptr == p + strlen(p));
  assert(errno == 0);

  p = "123a0";
  i = strtol(p, &endptr, 10);
  assert(i == 123);
  assert(endptr == p + 3);
  assert(errno == 0);

  p = "a123bc";
  i = strtol(p, &endptr, 10);
  assert(i == 0);
  assert(endptr == p);
  assert(errno == 0);  // note that errno is 0!
}

int main() {
  test_atoi();
  test_strtol();
  return 0;
}
