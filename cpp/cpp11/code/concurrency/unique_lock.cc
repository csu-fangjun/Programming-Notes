#include <cassert>
#include <mutex>

void test() {
  std::mutex m;
  {
    std::unique_lock<std::mutex> lock(m);
    assert((bool)lock == true);
    // it frees the mutex in the destructor
  }

  {
    std::unique_lock<std::mutex> lock(m, std::defer_lock_t());
    lock.lock();  // now lock the mutex
    assert((bool)lock == true);
    // the mutex is freed in the destructor
  }

  {
    m.lock();
    std::unique_lock<std::mutex> lock(m, std::adopt_lock_t());
    // we use adpot_lock() here since `m` has already been locked.
    assert((bool)lock == true);

    std::unique_lock<std::mutex> lock2 = std::move(lock);  // move constructible
    assert((bool)lock == false);
    assert(lock2.owns_lock() == true);
  }
}

void test2() {
  // lock mutliple mutexes
  std::mutex m1;
  std::mutex m2;
  std::unique_lock<std::mutex> lock1(m1, std::defer_lock_t());
  std::unique_lock<std::mutex> lock2(m2, std::defer_lock_t());
  assert(lock1.owns_lock() == false);
  assert(lock2.owns_lock() == false);

  std::lock(lock1, lock2);  // lock both mutex atomically

  assert(lock1.owns_lock() == true);
  assert(lock2.owns_lock() == true);
}

int main() {
  test();
  test2();
  return 0;
}
