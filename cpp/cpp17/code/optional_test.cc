#include <cassert>
#include <iostream>
#include <optional>
#include <string>

int main() {
  std::optional<std::string> s1("hello");
  std::optional<std::string> s2;
  s2 = std::move(s1);

  // note that s1 still has a value, though its value has been moved!
  assert((bool)s1 == true);
  assert(s1.has_value() == true);

  assert((bool)s2 == true);
  assert(s2.has_value() == true);

  assert((*s1).empty() == true);
  assert((*s2).empty() == false);

  assert(*s1 == "");
  assert(*s2 == "hello");

  assert(s1.value() == "");
  assert(s2.value() == "hello");

  // note that std::nullopt is an instance of class std::nullopt_t, which
  // is an empty class
  assert(sizeof(std::nullopt) == 1);
  assert(sizeof(std::nullopt_t) == 1);

  s2 = std::nullopt;
  assert(s2.has_value() == false);
  assert(s2.value_or("10") == "10");

  s2 = "123";
  assert(s2.has_value() == true);
  assert(*s2 == "123");
  assert(s2.value() == "123");
  assert(s2.value_or("10") == "123");

  s2 = {};
  assert(s2.has_value() == false);
}
