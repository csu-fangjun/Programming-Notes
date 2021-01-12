#include <cassert>
#include <iostream>
#include <random>

static void Test1() {
  // [10, 20]
  std::uniform_int_distribution<int32_t> dist(10, 20);

  std::mt19937 gen1(100); // 100 is the seed
  std::mt19937 gen2(100); // 100 is the seed

  int v1 = dist(gen1);
  int v2 = dist(gen2);
  assert(v1 == v2); // this is no state in uniform_int_distribution!
}

static void Test2() {
  std::mt19937 gen1(0);
  std::mt19937 gen2(0);
  std::uniform_int_distribution<int32_t> dist(0, 10);

  dist(gen1);
  gen1();
  dist(gen1);
  gen1();

  gen2.discard(4);
  assert(gen1() == gen2());
}

int main() {
  Test1();
  Test2();
  return 0;
}
