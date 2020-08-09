#include <future>
#include <iostream>

int add(int a, int b) { return a + b; }

int main() {
  {
    std::future<int> f = std::async(std::launch::async, add, 1, 2);
    // std::cout << "result: " << f.get() << "\n";
    // std::cout << "result: " << f.get() << "\n"; // exception!
  }
  return 0;
}
