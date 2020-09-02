#include <stdio.h>
#include <stdlib.h>

void f1() { printf("in f1\n"); }
void f2() { printf("in f2\n"); }

__attribute__((constructor)) void b2() { printf("in b2\n"); }
__attribute__((constructor)) void b1() { printf("in b1\n"); }
void b3() { printf("in b3 ctors\n"); }

__attribute__((section(".ctors"))) void (*b4)() = &b3;

void b5() { printf("in b5 dtors\n"); }
__attribute__((section(".dtors"))) void (*b6)() = &b5;

// like a stack, last registered functions are called first
void regiester_atexit() {
  atexit(&f1);
  atexit(&f2);
}

int main() {
  regiester_atexit();
  printf("exiting\n");
  // exit(0);  // return 0 or exit(0) are both OK!
  return 0;
}
