// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)

#include <cassert>

extern "C" {
int inc(int);
int get_val();
void set_val(int);
}

int main() {
  int a = inc(8);
  assert(a == 18);

  set_val(90);
  assert(get_val() == 90);
  a = inc(10);
  assert(a == 100);
  return 0;
}
