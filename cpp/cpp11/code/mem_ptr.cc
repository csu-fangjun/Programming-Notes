#include <iostream>

class A {
 public:
  virtual ~A() {}
  virtual void print() { std::cout << "A\n"; }
};

class B {
 public:
  virtual ~B() {}
  virtual void show() { std::cout << "B\n"; }
};

class C : public A, public B {
 public:
  void print() override { std::cout << "C\n"; }
  void show() override { std::cout << "B\n"; }
};

void test1() {
  A a;
  B b;
  C* c = (C*)&b;
  std::cout << "b: " << &b << "\n";  // 0x620
  std::cout << "c: " << c << "\n";   // 0x618
  B* b2 = c;
  std::cout << "b2: " << &b << "\n";  // 0x620

  c = (C*)&a;
  std::cout << "\na: " << c << "\n";  // 0x610
  std::cout << "c: " << c << "\n";    // 0x610
}
void test() { test1(); }

int main() {
  test();
  return 0;
}
