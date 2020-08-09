#include <type_traits>

struct Foo {
  using result_type = int;
};

template <typename...>
using my_void_t = void;

struct Bar {
  int hello;
};

namespace {

// this approache is more element
template <typename T, typename = void>
struct has_result_type : std::false_type {};

template <typename T>
struct has_result_type<T, my_void_t<typename T::result_type>> : std::true_type {

};

static_assert(has_result_type<Foo>::value, "");
static_assert(has_result_type<Bar>::value == false, "");

}  // namespace

namespace kk {

// this approach is deprecated.
template <typename T>
class has_result_type {
 private:
  struct true_t {
    char i[2];
  };
  struct false_t {
    char i;
  };

  template <typename U>
  static false_t __test(...);

  template <typename U>
  static true_t __test(typename U::result_type* = nullptr);

 public:
  static const bool value =
      sizeof(__test<T>(0)) == sizeof(true_t) ? true : false;
};

static_assert(has_result_type<Foo>::value, "");
static_assert(has_result_type<Bar>::value == false, "");
}  // namespace kk

namespace {

template <typename T, typename = void>
struct has_member_hello : std::false_type {};

template <typename T>
struct has_member_hello<T, my_void_t<decltype(std::declval<T>().hello)>>
    : std::true_type {};

static_assert(has_member_hello<Foo>::value == false, "");
static_assert(has_member_hello<Bar>::value, "");

}  // namespace

int main() { return 0; }
