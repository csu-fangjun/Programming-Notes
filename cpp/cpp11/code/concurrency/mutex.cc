#include <cassert>
#include <iostream>
#include <mutex>
#include <thread>

// note that std::mutex is not supposed to be used independently.
// We should use std::lock_guard, or std::unique_lock.

int g;
std::mutex m;

void test2() {
  // std::mutex m2(m); // compile time error, not copyable
  std::mutex m2;
  // m2 = m;  // compile time error, not assignable
}

void test() {
  for (int i = 0; i < 100000; ++i) {
    std::this_thread::yield();
    m.lock();  // it will block until other threads call unlock()
               // do not call lock() directly! use std::lock_guard or
               // std::unique_lock
    g += 1;
    m.unlock();
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
