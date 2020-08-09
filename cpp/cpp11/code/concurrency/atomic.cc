#include <unistd.h>

#include <atomic>  // we have include it to use std::atomic
#include <cassert>
#include <iostream>
#include <thread>
#include <type_traits>

// clang-format off
static void test_constructor() {
  {
    std::atomic<int> a;  // a is uninitialized
#if 0
    std::cout << a << "\n";  // a garbage value; it calls `operator int()` of
                             // std::atomic<int>
#endif
    // std::atomic<int> b(a);   // compile-time error; it is not copyable
    std::atomic<int> b;
    // b = a;  // compile time error, it is not assignable

    std::atomic_init(&a, 10);  // not thread safe. the passed object MUST be default constructed.
                               // it can be called on once for a given object.
                               // then why do we need it?
                               // it is used only for C compatiable.

    assert(a == 10);  // it uses the `operator int()` of std::atomic<int>
    a = 11;
    assert(a == 11);
  }
  {
    std::atomic<int> a(100);
    assert(a == 100);

    int b = a.exchange(10); // return the previous value atomically
    assert(b == 100);
    assert (a == 10);
    a += 1;
    assert(a == 11);
  }
}
// clang-format on

std::atomic<int> g;
void test() {
  for (int i = 0; i < 100000; ++i) {
    std::this_thread::yield();
    g += 1;  // no data race here since g is of type std::atomic<int>
  }
}

int main() {
  static_assert(std::is_same<std::atomic<bool>, std::atomic_bool>::value, "");
  static_assert(std::is_same<std::atomic<int>, std::atomic_int>::value, "");
  test_constructor();

  std::thread t1(test);
  std::thread t2(test);
  t1.join();
  t2.join();
  assert(g == 200000);
  return 0;
}
