#include <cstdint>
#include <iostream>
#include <type_traits>

static_assert(alignof(int32_t) == 4, "");
static_assert(alignof(float) == 4, "");

// see https://en.cppreference.com/w/cpp/types/alignment_of
static_assert(std::alignment_of<float>::value == 4, "");
static_assert(std::alignment_of<float>() == 4, "");
static_assert(std::alignment_of<float>{} == 4, "");

// need -std=c++14
// static_assert(std::alignment_of<float>{}() == 4, "");

// see https://en.cppreference.com/w/cpp/language/alignas
int32_t a;
static_assert(alignof(a) == 4, "");

alignas(8) int32_t b;
static_assert(alignof(b) == 8, "");

// see https://en.cppreference.com/w/cpp/types/aligned_storage

struct S {
  char a;
  int32_t b;
  S(char a, int32_t b) : a(a), b(b) { std::cout << "Constructor called\n"; }
  ~S() { std::cout << "Destructor called\n"; }
};

// note that std::aligned_storage is often used with placement new
// and destroyed with explicit destructor calls
struct Buf {
  typename std::aligned_storage<sizeof(S), alignof(S)>::type b[2];
};

int main() {
  Buf buf;
  new (&buf.b[0]) S(10, 20);
  new (&buf.b[1]) S(100, 200);

  std::cout << "[0]: a " << (int32_t)(reinterpret_cast<const S *>(&buf.b[0])->a)
            << "\n";
  std::cout << "[0]: b " << reinterpret_cast<const S *>(&buf.b[0])->b << "\n";

  std::cout << "[1]: a " << reinterpret_cast<const S *>(&buf.b[1])->a << "\n";
  std::cout << "[1]: b " << reinterpret_cast<const S *>(&buf.b[1])->b << "\n";

  reinterpret_cast<S *>(&buf.b[0])->~S();
  reinterpret_cast<S *>(&buf.b[1])->~S();
}
