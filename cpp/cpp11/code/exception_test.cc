#include <iostream>
// in libcxx
//  include/exception
//  src/runtime/exception/

int test() {
  try {
    throw std::exception{};
  } catch (const std::exception& ex) {
    std::cout << ex.what() << "\n";
  }
  // print:
  // std::exception
  //
  // the what() method is defined in
  // libcxx/src/support/runtime/exception_fallback.ipp
}

int main() {
  test();
  return 0;
}
