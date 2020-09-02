#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// setenv, unsetenv
// putenv, getenv
// clearenv

extern char **environ;
int print_all() {
  char **p = environ;
  while (p && *p) {
    printf("%s\n", *p);
    p++;
  }
  // it prints
  // name1=value1
  // name2=value2
  // ...
}

void test_getenv() {
  char *shell = getenv("SHELL");
  if (shell) {
    printf("shell is: %s\n", shell); // shell is: /bin/bash
  }

  char *no = getenv("no");
  if (no) {
    printf("no is: %s\n", no);
  } else {
    printf("environment variable no does not exist\n");
  }
}

void test_setenv() {
  const char *name = "hello";
  const char *value = "world";
  int overwrite = 1;

  char *b = getenv(name);
  assert(b == NULL);

  int ret = setenv(name, value, overwrite);
  assert(ret == 0);

  b = getenv(name);
  assert(b != NULL);
  assert(strcmp(b, value) == 0);

  ret = unsetenv(name);
  assert(ret == 0);

  b = getenv(name);
  assert(b == NULL);

  ret = unsetenv(name); // it is OK to delete a non-existed environment variable
  assert(ret == 0);
}

int main() {
  // print_all();
  test_getenv();
  test_setenv();
  return 0;
}
