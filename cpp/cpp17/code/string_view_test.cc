#include <cassert>
#include <cstring>
#include <iostream>
#include <type_traits>

#if defined(__GNUC__) && __GNUC__ < 7
#include <experimental/string_view>
#define string_view experimental::string_view
#else
#include <string_view>
#endif

// NOTE: string_view is usually **passed by value** in function calls

void test_size() {
  // it contains a poniter and a size_t
  static_assert(sizeof(std::string_view) == 16, "");
}

void test_constructor() {
  {
    std::string_view s;
    assert(s.empty() == true);
    assert(s.data() == nullptr);
    assert(s.size() == 0);
  }
  {
    const char *h = "1234567";
    std::string_view s(h, 3);
    assert(s.data() == h);
    assert(s.size() == 3);

    // copy constructor
    std::string_view s2(s);
    assert(s2.data() == s.data());
    assert(s2.size() == s.size());

    std::string_view s3(h);
    assert(s3.size() == 7);
    assert(s3.size() == strlen(h));
    assert(s3.data() == h);
  }
}

void test_iterator() {
  std::string_view s("1234");
  auto begin = s.begin(); // it is a const iterator! we cannot modify it!
  auto cbegin = s.cbegin();
  static_assert(std::is_same<decltype(begin), decltype(cbegin)>::value, "");

  // similar to end and cend

  static_assert(std::is_same<decltype(s[0]), const char &>::value, "");
  // s[0] = '1'; // compile error: assignment of read-only location

  static_assert(std::is_same<decltype(s.at(0)), const char &>::value, "");
  // s.at(0) = '1'; // compile error: assignment of read-only location

  static_assert(std::is_same<decltype(s.front()), const char &>::value, "");
  static_assert(std::is_same<decltype(s.back()), const char &>::value, "");
}

void test_prefix() {
  const char *h = "abcdef";
  std::string_view s(h);
  s.remove_prefix(2);
  assert(s.data() == h + 2);

  // now s is cdef

  s.remove_suffix(3);

  // now s is "c"
  assert(s.compare("c") == 0);
}

int main() {
  std::cout << "__cplusplus: " << __cplusplus << "\n";
  test_size();
  test_constructor();
  test_iterator();
  test_prefix();
}
