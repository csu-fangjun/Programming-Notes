#include <type_traits>

namespace k1 {

template <bool Head, bool... Tail>
struct Or : std::integral_constant<bool, Head || Or<Tail...>::value> {};

template <bool Head>
struct Or<Head> : std::integral_constant<bool, Head> {};

template <bool Head, bool... Tail>
struct And : std::integral_constant<bool, Head && And<Tail...>::value> {};

template <bool Head>
struct And<Head> : std::integral_constant<bool, Head> {};

void test() {
  static_assert(Or<true>::value, "");
  static_assert(Or<false>::value == false, "");
  static_assert(Or<true, false>::value, "");
  static_assert(Or<false, true, false>::value, "");

  static_assert(And<true>::value, "");
  static_assert(And<false>::value == false, "");
  static_assert(And<true, false>::value == false, "");
  static_assert(And<false, true, false>::value == false, "");
}

}  // namespace k1

namespace k2 {
template <bool Head, bool... Tail>
struct Or : std::conditional<Head, std::true_type, Or<Tail...>> {};

template <>
struct Or<false> : std::false_type {};

template <>
struct Or<true> : std::true_type {};

void test() {
  static_assert(Or<true>::value, "");
  static_assert(Or<false>::value == false, "");
  static_assert(Or<false, false>::value == false, "");
}
}  // namespace k2

int main() {
  k1::test();

  return 0;
}
