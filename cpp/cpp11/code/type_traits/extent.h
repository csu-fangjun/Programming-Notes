#pragma once

#include <cstdint>  // std::size_t

#include "integral_constant.h"

namespace kk {
template <typename T, std::size_t N = 0>
struct extent : integral_constant<std::size_t, 0> {};

template <typename T>
struct extent<T[], 0> : integral_constant<std::size_t, 0> {};

template <typename T, std::size_t N>
struct extent<T[], N> : extent<T, N - 1> {};

template <typename T, std::size_t I>
struct extent<T[I], 0> : integral_constant<std::size_t, I> {};

template <typename T, std::size_t I, std::size_t N>
struct extent<T[I], N> : extent<T, N - 1> {};

// note that we also need to provide the default value for N here
template <typename T, std::size_t N = 0>
inline constexpr std::size_t extent_v = extent<T, N>::value;

static_assert(extent_v<int> == 0);
static_assert(extent_v<int, 1> == 0);

static_assert(extent_v<int[]> == 0);
static_assert(extent_v<int[], 0> == 0);
static_assert(extent_v<int[], 1> == 0);

static_assert(extent_v<int[][3]> == 0);
static_assert(extent_v<int[][3], 1> == 3);
static_assert(extent_v<int[][3], 2> == 0);

static_assert(extent_v<int[3]> == 3);
static_assert(extent_v<int[3], 1> == 0);
static_assert(extent_v<int[3][5]> == 3);
static_assert(extent_v<int[3][5], 1> == 5);
static_assert(extent_v<int[3][5], 2> == 0);

}  // namespace kk
