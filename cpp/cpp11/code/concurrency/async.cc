#include <future>
#include <iostream>
#include <thread>

int add(int a, int b) { return a + b; }

int main() {
  std::future<int> f = std::async(add, 1, 2);
  std::cout << f.get() << "\n";
  return 0;
}
