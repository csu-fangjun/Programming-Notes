#pragma once

#include "integral_constant.h"
#include "remove_cv.h"
namespace kk {
#if 1
template <typename T>
struct is_pointer_helper : false_type {};

template <typename T>
struct is_pointer_helper<T *> : true_type {};

template <typename T>
struct is_pointer : is_pointer_helper<remove_cv_t<T>> {};

#else

// CAUTION: the following implementation is incorrect
// as it cannot handle the following cases:
//
// is_pointer<int *const>
// is_pointer<int *volatile>
//
template <typename T>
struct is_pointer : false_type {};

template <typename T>
struct is_pointer<T *> : true_type {};

#endif

template <typename T>
inline constexpr bool is_pointer_v = is_pointer<T>::value;

static_assert(is_pointer_v<int *> == true);
static_assert(is_pointer_v<const int *> == true);

static_assert(is_pointer_v<int *const> == true);     // CAUTION
static_assert(is_pointer_v<int *volatile> == true);  // CAUTION

// CAUTION: const char (&)[3]
static_assert(is_pointer_v<decltype("ab")> == false);

// CAUTION
static_assert(is_pointer_v<int (*)[]>);
static_assert(is_pointer_v<int (*)[3]>);
static_assert(is_pointer_v<int (*)()>);

namespace {
struct Foo {
  void Test(int) {}
  int m;
};

// CAUTION: member function pointer is false for is_pointer !!!
static_assert(is_pointer_v<decltype(&Foo::Test)> == false);

static_assert(is_pointer_v<void (Foo::*)(int)> == false);

static_assert(is_pointer_v<void (Foo::*const)(int)> == false);
static_assert(is_pointer_v<void (Foo::*)(int)> == false);

// CAUTION: data member pointer is false for is_pointer !!!
static_assert(is_pointer_v<decltype(&Foo::m)> == false);

static_assert(is_pointer_v<int Foo::*> == false);

// CAUTION
static_assert(is_pointer_v<std::nullptr_t> == false);
static_assert(is_pointer_v<decltype(nullptr)> == false);

}  // namespace

}  // namespace kk
