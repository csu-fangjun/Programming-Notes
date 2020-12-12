#include <cassert>
#include <iostream>
#include <regex>
#include <type_traits>

/* refer to

https://en.cppreference.com/w/cpp/regex/basic_regex

std::regex  == std::baisc_regex<char>
std::cmatch == std::match_results<const char*>
std::smatch == std::match_results<std::string::const_iterator>

std::match_results<> has a method `operator []` that returns
std::sub_match which can be converted to `std::string` implicitly.

meta character: dot .
literal character: a b c

*/

int main() {
  assert(std::is_trivial<std::regex>::value == 0);
  assert(std::is_trivially_default_constructible<std::regex>::value == 0);

  {
    std::regex re("a(a)*b");
    std::string target("aaab");
    std::smatch sm;
    assert(sm.size() == 0);
    assert(sm.empty() == true);

    std::regex_match(target, sm, re);
    assert(sm.size() == 2);
    assert(sm.empty() == false);
    assert(sm[0] == "aaab"); // 0: the entire match
    assert(sm[1] == "a");
    std::cout << "sm2: " << sm[0] << std::endl;
    // assert(sm[2] == "a");
  }
}
