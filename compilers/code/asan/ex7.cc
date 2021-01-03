// new[] + std::unique_ptr<>
// gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex7.cc -o ex7
#include <memory>

int main() {
  std::unique_ptr<int> p(new int[3]);
  return 0;
}
