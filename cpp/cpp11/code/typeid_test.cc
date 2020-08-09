// refer to
// https://en.cppreference.com/w/cpp/language/typeid
//
// To use `typeid`, we must include the header typeinfo
#include <cxxabi.h>  // for demangling, refer to pybind11

// refer to
// https://gcc.gnu.org/onlinedocs/libstdc++/libstdc++-html-USERS-4.3/a01696.html
// for abi::__cxa_demangle

#include <functional>
#include <iostream>
#include <memory>
#include <tuple>
#include <typeinfo>

template <typename T>
std::string GetTypeId() {
  std::string s = typeid(T).name();
  int status;
  std::unique_ptr<char, decltype(&std::free)> ptr{
      abi::__cxa_demangle(s.c_str(), nullptr, nullptr, &status), &std::free};

  if (status != 0) {
    std::cerr << "Failed to call __cxa_demangle\n";
    exit(1);
  }

  return std::string(ptr.get());
}

struct Foo {
  int a;
  void hello(int b, int c) {
    // clang-format off
    // std::cout << GetTypeId<decltype(world)>() << "\n"; // int* ()
    // std::cout << GetTypeId<decltype(*world)>() << "\n"; // int* ()
    // std::cout << GetTypeId<decltype(&world)>() << "\n"; // int* (*)()
    // std::cout << GetTypeId<decltype(Foo::world)>() << "\n"; // int* ()
    // std::cout << GetTypeId<decltype(&Foo::world)>() << "\n"; // int* (*)()
    // std::cout << GetTypeId<decltype(&Foo::hello)>() << "\n"; // void (Foo::*)(int, int)
    // clang-format on
  }
  static int* world() { return new int; }
};

namespace kk {
struct Bar {
  int hello() {}
};
namespace detail {
struct Cat {};
}  // namespace detail
}  // namespace kk

void test() {
#define P(x) std::cout << "typeid(" #x ").name(): " << typeid(x).name() << "\n"

  P(int8_t);           // a
  P(uint8_t);          // h
  P(int16_t);          // s
  P(uint16_t);         // t
  P(int32_t);          // i
  P(uint32_t);         // j
  P(float);            // f
  P(double);           // d
  P(test);             // FvvE
  P(&test);            // PFvvE
  P(Foo);              // 3Foo
  P(&Foo::a);          // M3Fooi
  P(&Foo::hello);      // M3FooFviiE
  P(kk::Bar);          // N2kk3BarE
  P(kk::detail::Cat);  // N2kk6detail3CatE

  // std::tuple<int, float> --> St5tupleIJifEE

  GetTypeId<kk::Bar>();                 // kk::Bar
  GetTypeId<int32_t>();                 // int
  GetTypeId<const int32_t>();           // int
  GetTypeId<kk::detail::Cat>();         // kk::detail::Cat
  GetTypeId<std::tuple<int, float>>();  // std::tuple<int, float>

  auto f1 = [](int i, float b) { return i + b; };
  auto f2 = [](float i, double b) { return i + b; };

  GetTypeId<decltype(f1)>();   // test()::{lambda(int float)#1}
  GetTypeId<decltype(f2)>();   // test()::{lambda(float, double)#2}
  GetTypeId<decltype(&f1)>();  // test()::{lambda(int float)#1}*
  // clang-format off
  GetTypeId<decltype(&decltype(f1)::operator())>(); // float (test()::{lambda(int, float)#1}::*)(int, float) const
  // clang-format on

  GetTypeId<std::function<void(int, int)>>();  // std::function<void (int,
                                               // int)>
  std::cout << "sizeof(std::function<void(int , int)>): "
            << sizeof(std::function<void(int, int)>) << "\n";

  GetTypeId<decltype(&kk::Bar::hello)>();  // int (kk::Bar::*)()

  std::cout << GetTypeId<decltype(&decltype(f1)::operator())>() << "\n";
  Foo().hello(1, 2);
}

int main() {
  test();
  return 0;
}
