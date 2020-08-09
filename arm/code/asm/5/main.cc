// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)

#include <iostream>

extern "C" {
const char* say_hello();
const char* say_world();
}

int main() {
  std::cout << say_hello() << " " << say_world() << "\n";
  return 0;
}
