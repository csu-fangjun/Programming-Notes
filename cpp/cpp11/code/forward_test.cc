
#include <iostream>
#include <type_traits>

namespace kk {

// if T is T, then return T&&
// if T is T&, then return T&
//
// Note t is always a lvalue reference, thus this function
// cannot accepts a function return value whose type is
// an rvalue
template <typename T>
T &&forward(typename std::remove_reference<T>::type &t) {
  std::cout << __PRETTY_FUNCTION__ << "\n";
  return static_cast<T &&>(t);
}

// the argument must be a rvalue reference
// So T cannot be a lvalue reference
template <typename T>
T &&forward(typename std::remove_reference<T>::type &&t) {
  static_assert(std::is_lvalue_reference<T>::value == false);
  std::cout << __PRETTY_FUNCTION__ << "\n";
  return static_cast<T &&>(t);
}

template <typename T>
void func(T &&t) {
  forward<T>(t);
}

void test() {
  int a = 1;

  // the T in func() is int&
  func(a);

  forward<int32_t>(10);
}

}  // namespace kk

int main() {
  kk::test();
  return 0;
}
