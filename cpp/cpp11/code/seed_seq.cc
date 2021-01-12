#include <cassert>
#include <iostream>
#include <random>

static void Test1() {
  std::seed_seq s;
  assert(s.size() == 0);
  std::vector<int32_t> v(3);

  // s generate is deterministic
  std::cout << "v\n";
  s.generate(v.begin(), v.end());
  for (auto i : v)
    std::cout << i << " ";
  std::cout << "\n";

  std::cout << "v2\n";
  std::vector<int32_t> v2(3);
  // `generate` is deterministic and depends only on
  // its internal `vector` and the size of the input iterator.
  s.generate(v2.begin(), v2.end());

  for (int i = 0; i != v2.size(); ++i) {
    assert(v[i] == v2[i]); // v and v2 are identical!
    std::cout << v2[i] << " ";
  }
  std::cout << "\n";

  std::vector<int32_t> v3(4);
  std::cout << "v3\n";
  s.generate(v3.begin(), v3.end());
  for (int i = 0; i != v3.size(); ++i) {
    if (i != v.size())
      assert(v[i] != v3[i]); // v3 is different since it has 4 values
    std::cout << v3[i] << " ";
  }
  std::cout << "\n";
}

void Test2() {
  std::seed_seq s{1, 2, 3};
  std::vector<int> seeds(5);
  s.generate(seeds.begin(), seeds.end()); // generate seeds
}

int main() {
  Test1();
  Test2();
  return 0;
}
