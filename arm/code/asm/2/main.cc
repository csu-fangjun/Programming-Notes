// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)

#include <cassert>

// inc is defined in inc.s
extern "C" int inc(int);

int main() {
  int a = inc(8);
  assert(a == 18);
  return 0;
}
