#pragma once

#include <cstdint>
#include <utility>  // for std::declval

#include "integral_constant.h"

namespace kk {

template <typename T, typename U>
struct is_same : false_type {};

template <typename T>
struct is_same<T, T> : true_type {};

// NOTE: how to define a variable template
// Requires C++17
template <typename T, typename U>
inline constexpr bool is_same_v = is_same<T, U>::value;

static_assert(is_same<int, int>::value);
static_assert(is_same<int, int *>::value == false);
static_assert(is_same<true_type, typename true_type::type>::value);
static_assert(is_same<true_type, typename false_type::type>::value == false);

static_assert(is_same_v<int, int> == true);
static_assert(is_same_v<int, int *> == false);

static_assert(is_same_v<int32_t, int>);

// CAUTION
static_assert(is_same_v<const char (&)[3], decltype("ab")>);

// for std::declval
static_assert(is_same_v<decltype(std::declval<int>()), int &&>);
static_assert(is_same_v<decltype(std::declval<int &>()), int &>);
static_assert(is_same_v<decltype(std::declval<int &&>()), int &&>);

static_assert(is_same_v<decltype(std::declval<const int>()), const int &&>);
static_assert(is_same_v<decltype(std::declval<const int &>()), const int &>);
static_assert(is_same_v<decltype(std::declval<const int &&>()), const int &&>);

}  // namespace kk
