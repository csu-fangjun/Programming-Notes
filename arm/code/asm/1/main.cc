// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)

#include <cassert>

// add is defined in add.s
//
// extern "C" is used to avoid name mangling
extern "C" int add(int, int);

int main() {
  int a = add(10, 20);
  assert(a == 30);
  return 0;
}
