#include <iostream>
#include <stdio.h>
#include <stdlib.h>

class Foo {
public:
  Foo() { std::cout << "this: " << this << std::endl; }
  ~Foo() { std::cout << "destructor called!\n"; }

  int a_;
};

int main() {
  int *a = (int *)malloc(sizeof(int));
  printf("a is %d, %p\n", *a, a);

  *a = 100;
  int *b = (int *)malloc(sizeof(int));
  printf("b is %d, %p\n", *b, b);

  free(b);
  free(a);

  std::cout << "----------\n";
  Foo *p = new Foo;
  std::cout << "address of p: " << p << std::endl;
  delete p;

  p = new Foo[3];
  std::cout << "address of p: " << p << std::endl;

  int *q = (int *)((char *)p - 8);
  std::cout << "num: " << *q << std::endl;
  *q = 1;

  delete[] p;

  return 0;
}
