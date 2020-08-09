#include <cassert>
#include <functional>  // for reference wrapper

int add(int a, int b) { return a + b; }

int main() {
  int a = 3;
  std::reference_wrapper<int> ra(a);
  // ra contains a int pointer to a
  // ra.get() return a reference to a
  //
  // get() is like std::unique_ptr<int>::get()
  assert(&ra.get() == &a);
  ra.get() = 5;
  assert(a == 5);
  assert(ra == 5);

  // ra can be implicitly converted to an int reference
  int& a2 = ra;
  assert(&a2 == &a);

  int b;
  std::reference_wrapper<int> rb(b);
  ra = rb;  // we can rebind the reference!
  assert(&ra.get() == &b);
  assert(&rb.get() == &b);

  // std::ref() return a std::reference_wrapper<T>
  // std::cref() return a std::reference_wrapper<const T>

  return 0;
}
