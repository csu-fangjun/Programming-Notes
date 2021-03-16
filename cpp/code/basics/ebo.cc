// g++ -o ebo ebo.cc

#include <cstdint>

namespace ex1 {

struct Base {};
static_assert(sizeof(Base) == 1, "");

struct Derived : Base {
  int32_t a;
};

// because of empty base optimization, its size is sizeof(int32_t)
static_assert(sizeof(Derived) == 4, "");

} // namespace ex1

namespace ex2 {

struct Base1 {};
struct Base2 {};
struct Derived : Base1, Base2 {
  int32_t i;
};

static_assert(sizeof(Base1) == 1, "");
static_assert(sizeof(Base2) == 1, "");

// Note that for multiple inheritance, Base1
// and Base2 do not occupy space here!
static_assert(sizeof(Derived) == 4, "");

} // namespace ex2

namespace ex3 {
struct Base1 {
  int32_t a;
};
struct Base2 {};

struct Derived1 : Base1, Base2 {
  int32_t b;
};

struct Derived2 : Base2, Base1 {
  int32_t b;
};

static_assert(sizeof(Base1) == 4, "");
static_assert(sizeof(Base2) == 1, "");

// note that EBO is independent on the inheritance order!
static_assert(sizeof(Derived1) == 8, "");
static_assert(sizeof(Derived2) == 8, "");

} // namespace ex3

namespace ex4 {
struct Base {};
struct Foo {
  Base b; // or place it after `a`, the result is the same
  int a;
};

static_assert(sizeof(Base) == 1, "");

// If an empty class is used as a data member, EBO is not applicable here!
static_assert(sizeof(Foo) == 8, "");

} // namespace ex4

namespace ex5 {}

int main() { return 0; }
