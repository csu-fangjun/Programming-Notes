#include <cassert>
#include <cstdlib>
#include <iostream>

static void Test1() {
  srand(100);
  int v1 = rand();
  int v2 = rand();
  assert(v1 != v2);
  assert(v1 != rand());

  srand(100);
  assert(v1 == rand());
  assert(v2 == rand());
}

static void Test2() {
  srand(10);
  int v1 = rand();

  srand(10);
  unsigned int seed = 100;
  int v2 = rand_r(&seed);
  assert(rand() == v1); // rand_r() did not call `rand()`!

  assert(seed != 100); // seed is changed inside `seed_r()`
}

int main() {
  Test1();
  Test2();
  return 0;
}
