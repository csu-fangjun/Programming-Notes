#include <iostream>
#include <sstream>
#include <thread>

thread_local int g_a = 10;

static void Test(int i) {
  std::ostringstream os;
  os << "id: " << i << ", g_a " << g_a << "\n";
  g_a += 1;
  os << "id: " << i << ", g_a " << g_a << "\n";
  g_a += 1;
  os << "id: " << i << ", g_a " << g_a << "\n";
  std::cout << os.str();
}

struct Foo {
  int i = 0;
};

static void TestPointer(Foo *f) { f->i += 10; }

static void test1() {
  std::thread t1(Test, 1);
  std::thread t2(Test, 2);
  t1.join();
  t2.join();
}

static void test2() {
  Foo f;
  std::thread t(TestPointer, &f);
  t.join();
  std::cout << "f.i is: " << f.i << std::endl;
}

int main() {
  // test1();
  // test2();
  return 0;
}
