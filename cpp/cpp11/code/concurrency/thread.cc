#include <cassert>
#include <iostream>
#include <thread>

static void add(int a, int b) {
  std::cout << __FUNCTION__ << ": " << a << " " << b << "\n";
}

static void test_add() {
  // pass parameters 1 and 2 to add
  std::thread thread(add, 1, 2);  // the thread starts run automatically
  thread.join();
}

static void pointer_arg(int* p) { *p += 2; }
static void test_pointer_arg() {
  int a = 10;
  std::thread thread(pointer_arg, &a);
  thread.join();
  std::cout << __FUNCTION__ << ": after thread: " << a << "\n";
  assert(a == 12);
}

static void ref_arg(int& p) { p += 2; }
// clang-format off
static void test_ref_arg() {
  int a = 10;
  std::thread thread(ref_arg, std::ref(a));  // note that we have to use std::ref
  thread.join();
  std::cout << __FUNCTION__ << ": after thread: " << a << "\n";
  assert(a == 12);
}

static void cref_arg(const int& p) {}
static void test_cref_arg() {
  int a = 10;
  std::thread thread(cref_arg, a); // for const int&, we can pass a!
  // std::thread thread(cref_arg, 10); // for const int&, we can pass a temporary variable!
  thread.join();
  std::cout << __FUNCTION__ << ": after thread: " << a << "\n";
  assert (a == 10);
}

class Foo{
  public:
  void bar() const {std::cout << "a_ is: " << a_ << "\n";}
  private:
  int a_ = 10;

};

static void test_class_method() {
  Foo f;
  std::thread t(&Foo::bar, &f); // we have to pass a `this` pointer
  t.join();
}

// clang-format on

int main() {
  // test_add();
  // test_pointer_arg();
  // test_ref_arg();
  // test_cref_arg();
  test_class_method();

  return 0;
}
