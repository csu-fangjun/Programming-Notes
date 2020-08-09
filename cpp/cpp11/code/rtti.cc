#include <typeinfo>

class A {
 public:
  int a;
};

class B : public A {
 public:
  int b;
};

int main() {
  B b;
  A* a = &b;
  b = dynamic_cast<B*>(a);
  // a = dynamic_cast<A*>(&b);
  return 0;
}
