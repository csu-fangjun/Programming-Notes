#include <cassert>
#include <mutex>
#include <thread>

int g;
std::mutex m;

void test() {
  for (int i = 0; i < 100000; ++i) {
    // it calls m.lock() in the constructor
    // and m.unlock() in the destructor
    //
    // this may not the base use cases for std::lock_guard
    // as it is called inside the for loop.
    std::lock_guard<std::mutex> lock(m);
    g += 1;
  }
}

int main() {
  std::thread t1(test);
  std::thread t2(test);
  t1.join();
  t2.join();
  assert(g == 200000);
  return 0;
}
