#include <cassert>
#include <numeric>
#include <vector>

int main() {
  std::vector<int> a(10);
  std::iota(a.begin(), a.end(), 0);
  for (std::size_t i = 0; i != a.size(); ++i) {
    assert(a[i] == i);
  }
  return 0;
}
