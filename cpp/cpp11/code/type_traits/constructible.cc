#include <type_traits>

struct A0 {};  // is trivial

struct A1 {
  A1();  // is not trivial!
};

struct A2 {  // is not tirival!
  A2() {}
};

struct A3 {  // is tirival
  A3() = default;
  int a;
};

struct A4 {  // is trivial
  int a;
};

struct A5 {
  int a = 1;  // not trivial since a is initialized
};

struct A6 {  // is trivial
  int &a;
};

struct A7 {  // is not trivial
  int &a;
  virtual void ABC() {}
};

struct A8 {
  A8() = default;
  A8(int a) : a(a) {}
  int a;
};

struct A9 {
  A9(const A9 &rhs) { a = rhs.a; }
  int a;
};

struct A10 {
  A10(const A10 &) = default;
  int a;
};

static void TestIsTrivial() {
  static_assert(std::is_trivial_v<int>);
  static_assert(std::is_trivial_v<int *>);
  static_assert(std::is_trivial_v<int (*)(int)>);
  static_assert(std::is_trivial_v<int[]>);

  static_assert(std::is_trivial_v<int &> == false);  // CAUTION

  static_assert(std::is_trivial_v<A0>);
  static_assert(std::is_trivial_v<A1> == false);
  static_assert(std::is_trivial_v<A2> == false);
  static_assert(std::is_trivial_v<A3>);
  static_assert(std::is_trivial_v<A4>);
  static_assert(std::is_trivial_v<A5> == false);
  static_assert(std::is_trivial_v<A6>);
  static_assert(std::is_trivial_v<A7> == false);

  static_assert(std::is_trivially_constructible_v<A0>);
  static_assert(std::is_trivially_constructible_v<A4, int> == false);  // why
  static_assert(std::is_trivially_constructible_v<A5, int> == false);  // why

  static_assert(std::is_trivial_v<A8>);  // CAUTION!
  static_assert(std::is_constructible_v<A8, int>);
  static_assert(std::is_trivially_constructible_v<A8, int> == false);

  // if we define a copy constructor for A8, then it is no longer
  // is_trivially_constructible !!!
  static_assert(std::is_trivially_constructible_v<A8, A8>);
  static_assert(std::is_trivially_constructible_v<A8, A8 &>);
  static_assert(std::is_trivially_constructible_v<A8, A8 &&>);

  static_assert(std::is_trivially_constructible_v<int>);
  static_assert(std::is_trivially_constructible_v<int, int>);
  static_assert(std::is_trivially_constructible_v<int, int &>);
  static_assert(std::is_trivially_constructible_v<int, int &&>);

  static_assert(std::is_trivial_v<A9> == false);
  static_assert(std::is_constructible_v<A9, A9>);
  static_assert(std::is_constructible_v<A9, A9 &>);
  static_assert(std::is_constructible_v<A9, A9 &&>);

  // the copy constructor of A9 is not trivial
  static_assert(std::is_trivially_constructible_v<A9, A9> == false);
  static_assert(std::is_trivially_constructible_v<A9, A9 &> == false);
  static_assert(std::is_trivially_constructible_v<A9, A9 &&> == false);

  // the copy constructor of A10 is trivial (using defaulted)
  static_assert(std::is_trivially_constructible_v<A10, A10>);
  static_assert(std::is_trivially_constructible_v<A10, A10 &>);
  static_assert(std::is_trivially_constructible_v<A10, A10 &&>);
}

class B0 {};
class B1 {
  int a;
};
class B2 {  // trivially copyable
  B2() {}
  int a;
};

class B3 {
  B3(const B3 &) = default;  // trivially copyable
  int a;
};

class B4 {
  B4(const B4 &) = default;  // trivially copyable
};

// it has a user provided copy constructor, so
// it is no longer trivially copyable, even
// if the function is empty
class B5 {
  B5(const B5 &) {}  // not trivially copyable
};

static void TestIsTriviallyCopyable() {
  static_assert(std::is_trivially_copyable_v<B0>);
  static_assert(std::is_trivially_copyable_v<B1>);
  static_assert(std::is_trivially_copyable_v<B2>);
  static_assert(std::is_trivially_copyable_v<B3>);
  static_assert(std::is_trivially_copyable_v<B4>);
  static_assert(std::is_trivially_copyable_v<B5> == false);
  //
}

int main() {
  TestIsTrivial();
  TestIsTriviallyCopyable();
  return 0;
}
