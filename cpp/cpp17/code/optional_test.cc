#include <cassert>
#include <iostream>
#include <optional>
#include <string>

struct S {
  S() { std::cout << "constructor called\n"; }
  S(const S &) { std::cout << "copy constructor called\n"; }
  S &operator=(const S &) {
    std::cout << "copy assignment operator called\n";
    return *this;
  }

  S(S &&) { std::cout << "move constructor called\n"; }
  S &operator=(S &&) {
    std::cout << "move assignment operator called\n";
    return *this;
  }

  ~S() { std::cout << "destructor called\n"; }
};

void test() {
  {
    std::optional<S> op; // print nothing
  }

  {
    std::optional<S> op;
    op = S{};
    // constructor called
    // move constructor called
    // destructor called
    // destructor called
  }
  std::cout << "--end case2\n";
  {
    std::optional<S> op(std::in_place);
    // constructor called
    // destructor called
  }
  std::cout << "--end case3\n";
  {
    S s;
    std::optional<S> op(s);
    // constructor called
    // copy constructor called
    // destructor called
    // destructor called
  }
  std::cout << "--end case4\n";
  {
    S s;
    std::optional<S> op(std::move(s));
    // constructor called
    // move constructor called
    // destructor called
    // destructor called
  }
  std::cout << "--end case5\n";
  {
    std::optional<S> op1(S{});
    std::optional<S> op2(S{});
    op1 = op2;
    // constructor called
    // move constructor called
    // destructor called
    //
    // constructor called
    // move constructor called
    // destructor called
    //
    // copy assignment operator called
    // destructor called
    // destructor called
  }
  std::cout << "--end case6\n";
  {
    std::optional<S> op1;
    std::optional<S> op2(S{});
    op1 = std::move(op2);
    assert(op1.has_value());
    assert(op2.has_value());
    // constructor called
    // move constructor called
    // destructor called
    //
    // move constructor called
    // destructor called
    // destructor called
  }
  std::cout << "--end case7\n";
  {
    std::optional<S> op;
    op.emplace();
    // constructor called
    // destructor called
  }
}

void test0() {
  std::optional<int32_t> op;
  assert(!op);
  assert(op == std::nullopt);
  assert(op.has_value() == false);
  assert(op.value_or(-10) == -10);
}

void test1() {
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
int main() {
  test();
  test0();
  test1();
  return 0;
}
