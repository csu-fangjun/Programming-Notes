#include <cstdint>
#include <iostream>
#include <type_traits>
namespace {

template <std::size_t... S>
struct index_sequence {};

template <std::size_t N, std::size_t... S>
struct index_sequence_impl : index_sequence_impl<N - 1, N - 1, S...> {};

template <std::size_t... S>
struct index_sequence_impl<0, S...> {
  using type = index_sequence<S...>;
};

template <std::size_t N>
using make_index_sequence = typename index_sequence_impl<N>::type;

template <std::size_t... S>
auto reverse_impl(index_sequence<S...>)
    -> index_sequence<sizeof...(S) - 1 - S...>;

template <std::size_t N>
using reverse_index_sequence = decltype(reverse_impl(make_index_sequence<N>()));

template <bool B>
using bool_constant = std::integral_constant<bool, B>;

template <bool...>
struct swallow {};
template <typename... T>
struct all_of
    : std::is_same<swallow<T::value..., true>, swallow<true, T::value...>> {};

void test() {
  reverse_index_sequence<3> s;
  std::cout << all_of<std::true_type, std::false_type>::value << std::endl;
  std::cout << all_of<std::true_type, std::true_type>::value << std::endl;
}

}  // namespace

int main() {
  test();
  return 0;
}
