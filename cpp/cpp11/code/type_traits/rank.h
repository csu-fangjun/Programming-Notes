#pragma once

#include <cstdint>  // for std::size_t

#include "integral_constant.h"

namespace kk {

template <typename T>
struct rank : integral_constant<std::size_t, 0> {};

template <typename T>
struct rank<T[]> : integral_constant<std::size_t, rank<T>::value + 1> {};

template <typename T, std::size_t N>
struct rank<T[N]> : integral_constant<std::size_t, rank<T>::value + 1> {};

template <typename T>
inline constexpr std::size_t rank_v = rank<T>::value;

static_assert(rank_v<int> == 0);
static_assert(rank_v<int[]> == 1);
static_assert(rank_v<int[2]> == 1);

// it first expands rank_v<int[][2]> -> rank_v<int[2]> -> rank_v<int>
// Note that int[][2] becomes int[2].
//
// That is,
// struct rank<int[][2]> : rank<int[2]>
// struct rank<int[2]> : rank<int>
static_assert(rank_v<int[][2]> == 2);

// The inheritance relationship is
// struct rank<int[][3][10]> : rank<int[3][10]>
// struct rank<int[3][10]> : rank<int[10]>
// struct rank<int[10]> : rank<int>
static_assert(rank_v<int[][3][10]> == 3);

}  // namespace kk
