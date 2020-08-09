// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)

#include <unistd.h>

#include <cstring>
#include <iostream>

#include "ruy/profiler/instrumentation.h"
#include "ruy/profiler/profiler.h"

void test2() {
  ruy::profiler::ScopeLabel my_label("test2");
  std::unique_ptr<int[]> p1(new int[1000]);
  std::unique_ptr<int[]> p2(new int[1000]);

  for (int i = 0; i < 1000; ++i) {
    memcpy(p1.get(), p2.get(), 1000 * sizeof(int));
  }
  sleep(1);
}

void test1(int d) {
  ruy::profiler::ScopeLabel t("test1: %d", d);
  std::unique_ptr<int[]> p1(new int[1000]);
  std::unique_ptr<int[]> p2(new int[1000]);

  {
    ruy::profiler::ScopeLabel t("inner_before_sleep: %d", d);
    sleep(1);
    ruy::profiler::ScopeLabel tt("inner_after_sleep: %d", d);
    for (int i = 0; i < 1000; ++i) {
      memcpy(p1.get(), p2.get(), 1000 * sizeof(int));
    }
  }
  {
    ruy::profiler::ScopeLabel t("inner2: %d", d);
    sleep(2);
  }

  test2();
}

int main() {
  ruy::profiler::ScopeProfile profile("main");
  ruy::profiler::ScopeLabel t("main");
  sleep(2);

  test1(10);
  test2();
  test1(100);

  return 0;
}
