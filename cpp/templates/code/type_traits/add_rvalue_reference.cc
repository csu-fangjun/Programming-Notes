#include <type_traits>

// Note that the type returned
// by `add_rvalue_reference` is `T&&`,
// which is a left value reference when `T` is `T&`

struct S {};

#define STATIC_ASSERT(p) static_assert((p), "")

int main() {
  STATIC_ASSERT(
      std::is_rvalue_reference<std::add_rvalue_reference_t<S>>::value);

  STATIC_ASSERT(
      std::is_lvalue_reference<std::add_rvalue_reference_t<S&>>::value);

  STATIC_ASSERT(
      std::is_rvalue_reference<std::add_rvalue_reference_t<S&&>>::value);

  STATIC_ASSERT(std::is_rvalue_reference<decltype(std::declval<S>())>::value);

  STATIC_ASSERT(std::is_lvalue_reference<decltype(std::declval<S&>())>::value);

  STATIC_ASSERT(std::is_rvalue_reference<decltype(std::declval<S&&>())>::value);

  return 0;
}
