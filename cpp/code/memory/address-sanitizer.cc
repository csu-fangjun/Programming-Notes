//
// refer to https://github.com/google/sanitizers/wiki/AddressSanitizer
//
// compile it using
//
//   gcc -fsanitize=address address-sanitizer.cc
//   ./a.out
//
// Add -g to show the line number and filename in the diagnostic output
// Add -O1 or other optimization flags so that it runs faster.
//
// It will link to libasan.so

#include <memory>
#include <stdlib.h>

void NoFree() {
  { void *p = malloc(1); }
}

void NoFreeCppNew() { void *p = new int; }

// disable address sanitizer for this function
__attribute__((no_sanitize_address)) int UseAfterFree() {
  int *p = reinterpret_cast<int *>(malloc(3 * sizeof(int)));
  free(p);
  return p[2];
}

int TestUniquePtr1() { std::unique_ptr<int[]> p(new int); }
int TestUniquePtr2() { std::unique_ptr<int> p(new int[1]); }

int main() {
  // NoFreeCppNew();
  // NoFree();
  // UseAfterFree();
  // TestUniquePtr1();
  // TestUniquePtr2();
  return 0;
}
