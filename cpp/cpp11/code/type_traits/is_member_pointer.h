#pragma once
#include "integral_constant.h"
#include "remove_cv.h"

namespace kk {

#if 1
template <typename T>
struct is_member_pointer_helper : false_type {};

template <typename T, typename U>
struct is_member_pointer_helper<T U::*> : true_type {};

template <typename T>
struct is_member_pointer : is_member_pointer_helper<remove_cv_t<T>> {};

#else

// the following implementation is incorrect as it cannot handle
// static_assert(is_member_pointer_v<int FooIsMemberPointer::*const> == true);
// static_assert(is_member_pointer_v<int FooIsMemberPointer::*volatile> ==
// true);

template <typename T>
struct is_member_pointer : false_type {};

template <typename T, typename U>
struct is_member_pointer<T U::*> : true_type {};

#endif

template <typename T>
inline constexpr bool is_member_pointer_v = is_member_pointer<T>::value;

namespace {

struct FooIsMemberPointer {
  void Test(int) {}
  int m;
};

static_assert(is_member_pointer_v<decltype(&FooIsMemberPointer::m)>);

// CAUTION
static_assert(is_member_pointer_v<decltype(&FooIsMemberPointer::Test)>);
static_assert(is_member_pointer_v<decltype(&FooIsMemberPointer::m)>);

// CAUTION: it can be either a data member or a member function!
static_assert(is_member_pointer_v<int FooIsMemberPointer::*>);
static_assert(is_member_pointer_v<void (FooIsMemberPointer::*)(int)>);

// CAUTION: it must use the helper
static_assert(is_member_pointer_v<int FooIsMemberPointer::*const> == true);
static_assert(is_member_pointer_v<int FooIsMemberPointer::*volatile> == true);

// CAUTION: raw pointers are not member pointers!
static_assert(is_member_pointer_v<int> == false);
static_assert(is_member_pointer_v<int *> == false);
static_assert(is_member_pointer_v<int (*)[]> == false);
static_assert(is_member_pointer_v<int (*)()> == false);

static_assert(is_member_pointer_v<decltype(nullptr)> == false);

}  // namespace

}  // namespace kk
