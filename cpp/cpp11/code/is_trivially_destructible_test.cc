#include <type_traits>

class Foo {};

class Bar {
public:
  ~Bar() {} // no
};

class Baz {
public:
  ~Baz() = default; // yes
};

class FooBaz {
public:
  ~FooBaz(); // no
};

FooBaz::~FooBaz() = default; // even if it is `default`, but its first
                             // declaration does not specify this!

int main() {
  static_assert(std::is_trivially_destructible<Foo>::value, "");
  static_assert(std::is_trivially_destructible<Bar>::value == false, "");
  static_assert(std::is_trivially_destructible<Baz>::value, "");
  static_assert(std::is_trivially_destructible<FooBaz>::value == false, "");
  return 0;
}
