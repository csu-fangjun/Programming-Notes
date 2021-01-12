#include <cassert>
#include <iostream>
#include <random>

static void Test1() {
  std::mt19937 gen; // gen is short for generator
  for (int i = 0; i != 9999; ++i)
    gen();
  assert(gen() == 4123659995); // the 10000th number is 4123659995
}

static void Test2() {
  std::mt19937 gen1(0); // the first way to set the seed
  std::mt19937 gen2(0); // set the seed to 0

  assert(gen1() == gen2());
  gen1();

  assert(gen1() != gen2());
  gen2.discard(1); // equivalent to gen2() and the discard the return value
  assert(gen1() == gen2());

  gen1.seed(100); // the second way to set the seed
  gen2.seed(100);
  assert(gen1() == gen2());

  std::seed_seq seq{10, 20, 30};
  gen1.seed(seq); // the third way to set the seed
  gen2.seed(seq);
  assert(gen1() == gen2());
}

static void Test3() {
  std::random_device rd;
  std::mt19937 gen(rd()); // rd() returns a random integer from /dev/urandom
                          // and it is used as a seed!

  int v = gen();

  gen.seed(10);
  int32_t v2 = gen();

  gen.seed(10);
  int32_t v3 = gen();

  assert(v != v2);
  assert(v2 == v3);
}

static void Test4() {
  std::seed_seq seq{0};
  std::mt19937 gen1(seq); // init with a std::seed_seq

  std::mt19937 gen2(0); // init with a integer seed
  std::mt19937 gen3;    // default initialized

  int32_t v1 = gen1();
  int32_t v2 = gen2();
  int32_t v3 = gen3();

  assert(v1 != v2);
  assert(v1 != v3);
  assert(v2 != v3);
}

static void Test5() {
  std::mt19937 gen1;
  std::mt19937 gen2(5489);
  assert(std::mt19937::default_seed == 5489);
  assert(gen1() == gen2());
}

int main() {
  Test1();
  Test2();
  Test3();
  Test4();
  Test5();

  return 0;
}
