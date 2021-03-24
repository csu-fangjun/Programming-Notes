#pragma once

#include "integral_constant.h"
#include "is_same.h"

namespace kk {
template <typename T>
struct is_const : false_type {};

template <typename T>
struct is_const<const T> : true_type {};

template <typename T>
inline constexpr bool is_const_v = is_const<T>::value;

static_assert(is_const_v<int> == false);
static_assert(is_const_v<int *> == false);

static_assert(is_const_v<const int>);
static_assert(is_const_v<int *const>);
static_assert(is_const_v<const int *> == false);  // CAUTION
static_assert(is_const_v<const int *const>);

static_assert(is_const_v<decltype("ab")> == false);  // CAUTION
static_assert(is_same_v<decltype("ab"), const char (&)[3]>);
static_assert(is_const_v<const char (&)[]> == false);  // CAUTION
static_assert(is_const_v<const char (*)[]> == false);  // CAUTION
static_assert(is_const_v<const char (*const)[]>);      // CAUTION
static_assert(is_const_v<const char[]>);               // CAUTION
static_assert(is_const_v<int (*)()> == false);         // CAUTION
static_assert(is_const_v<int (*const)()>);             // CAUTION

}  // namespace kk
