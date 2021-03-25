#pragma once

#include <cstdint>  // std::size_t

#include "is_same.h"
#include "remove_reference.h"

namespace kk {

template <typename T>
struct remove_extent {
  using type = T;
};

template <typename T>
struct remove_extent<T[]> {
  using type = T;
};

template <typename T, std::size_t N>
struct remove_extent<T[N]> {
  using type = T;
};

template <typename T>
using remove_extent_t = typename remove_extent<T>::type;

static_assert(is_same_v<int, remove_extent_t<int>>);
static_assert(is_same_v<int, remove_extent_t<int[]>>);
static_assert(is_same_v<int, remove_extent_t<int[3]>>);

static_assert(is_same_v<int[3], remove_extent_t<int[][3]>>);
static_assert(is_same_v<int[2], remove_extent_t<int[1][2]>>);
static_assert(is_same_v<int[5][6], remove_extent_t<int[8][5][6]>>);

// CAUTION
static_assert(is_same_v<const char (&)[3], remove_extent_t<decltype("ab")>>);

// CAUTION
static_assert(
    is_same_v<const char, remove_extent_t<remove_reference_t<decltype("ab")>>>);

}  // namespace kk
