#include <iostream>
#include <type_traits>

namespace {

template <typename T>
void add(T a,
         typename std::enable_if<std::is_integral<T>::value>::type* = nullptr) {
  std::cout << "integral add\n";
}

template <typename T>
void add(T a, typename std::enable_if<std::is_floating_point<T>::value>::type* =
                  nullptr) {
  std::cout << "float add\n";
}

void test() {
  int a[3];
  std::cout << "int a[3] is array: " << std::is_array<decltype(a)>::value
            << std::endl;

  const int b[1] = {1};
  std::cout << "const int b[3] is array: " << std::is_array<decltype(b)>::value
            << std::endl;

  std::cout << "is function: free: " << std::is_function<decltype(free)>::value
            << std::endl;

  std::cout << "is function: &free: "
            << std::is_function<decltype(&free)>::value << std::endl;
  add(1);
  add(1.0);
}

void test2() {
  // add pointer
  static_assert(std::is_same<std::add_pointer<int>::type, int*>::value, "");
  static_assert(std::is_same<std::add_pointer<int&>::type, int*>::value, "");

  // remove pointer
  static_assert(std::is_same<std::remove_pointer<int*>::type, int>::value, "");
  static_assert(std::is_same<std::remove_pointer<int>::type, int>::value, "");

  // std::decay
  static_assert(std::is_same<std::decay<int>::type, int>::value, "");
  static_assert(std::is_same<std::decay<int[3]>::type, int*>::value,
                "");  // array to poniter
  static_assert(std::is_same<std::decay<int&>::type, int>::value, "");
  static_assert(std::is_same<std::decay<const int&>::type, int>::value, "");
  static_assert(std::is_same<std::decay<int*>::type, int*>::value, "");
  static_assert(std::is_same<std::decay<const int*>::type, const int*>::value,
                "");

  // a function pointer with std::decay
  static_assert(
      std::is_same<std::decay<decltype(test)>::type, decltype(&test)>::value,
      "");

  // std::remove_cv
  static_assert(std::is_same<std::remove_cv<const int>::type, int>::value, "");
  static_assert(
      std::is_same<std::remove_cv<const int*>::type, const int*>::value, "");

  static_assert(std::is_same<std::remove_cv<int* const>::type, int*>::value,
                "");

  // std::result_of
  static_assert(
      std::is_same<std::result_of<decltype(test)*()>::type, void>::value, "");
}

}  // namespace

int main() {
  test();
  return 0;
}
