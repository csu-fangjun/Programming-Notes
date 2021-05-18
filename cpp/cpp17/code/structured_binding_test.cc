
#include <cassert>
#include <cstdint>
#include <type_traits>

// array
void test1() {
  int32_t arr[2] = {1, 2};

  // a temp array e is created and arr
  // is copied to e
  //
  // a refers to e[0]
  // b refers to e[1]
  auto [a, b] = arr;
  static_assert(std::is_same_v<decltype(a), int32_t>);
  static_assert(std::is_same_v<decltype(b), int32_t>);

  assert(&a != &arr[0]);
  assert(&b != &arr[1]);

  a = 10;
  b = 20;

  // arr is not changed
  assert(arr[0] == 1);
  assert(arr[1] == 2);
}

void test2() {
  int32_t arr[2] = {1, 2};

  // the temp array e is a reference to arr,
  //
  // a is an alias to e[0], which is arr[0]
  auto &[a, b] = arr;
  static_assert(std::is_same_v<decltype(a), int32_t>);
  static_assert(std::is_same_v<decltype(b), int32_t>);

  assert(&a == &arr[0]);
  assert(&b == &arr[1]);

  a = 10;
  b = 20;
  assert(arr[0] == 10);
  assert(arr[1] == 20);
}

void test3() {
  int32_t arr[2] = {1, 2};

  // the temp array e is a reference to arr,
  //
  // a is an alias to e[0], which is arr[0]
  //
  // since e is a const array, so both
  // a and b are const
  const auto &[a, b] = arr;
  static_assert(std::is_same_v<decltype(a), const int32_t>);
  static_assert(std::is_same_v<decltype(b), const int32_t>);

  assert(&a == &arr[0]);
  assert(&b == &arr[1]);

  // a = 10;  // error: assignment of read-only variable 'a'
  // b = 20;  // error: assignment of read-only variable 'b'
}

void test4() {
  int32_t arr[2] = {1, 2};
  // the temp array e is const
  // so both a and b are const
  //
  // arr is copied to e
  const auto [a, b] = arr;
  static_assert(std::is_same_v<decltype(a), const int32_t>);
  static_assert(std::is_same_v<decltype(b), const int32_t>);
  // a = 10;  // error: assignment of read-only variable 'a'
  // b = 20;  // error: assignment of read-only variable 'b'

  assert(&a != &arr[0]);
  assert(&b != &arr[1]);
}

int main() {
  test1();
  test2();
  test3();
  test4();
  return 0;
}
