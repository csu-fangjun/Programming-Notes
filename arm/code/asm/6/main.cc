// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)

#include <cassert>

extern "C" {
int test();
}

int main() {
  int a = test();
  assert(a == 10);
  return 0;
}
