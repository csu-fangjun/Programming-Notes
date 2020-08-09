#include <cassert>

#include "abc/addressbook.pb.h"

using kk::Person;

void Test() {
  Person p;
  p.set_name("Tom");
  p.set_id(10);
  std::cout << p.DebugString();
}

int main() {
  Test();
  return 1;
}
