#include <iostream>
#include <memory>

int main() {
  std::cout << "size of unique_ptr: " << sizeof(std::unique_ptr<int>) << "\n";

  int a;
  int b[10];
  auto f = [b](void *p) {};
  std::unique_ptr<int, decltype(f)> p(
      nullptr, f); // since f has captures, the size of p is more than a pointer
  std::cout << "size of p: " << sizeof(decltype(p)) << "\n";

  std::cout << "size of shared_ptr: " << sizeof(std::shared_ptr<int>)
            << "\n"; // 16
  return 0;
}
