#pragma once
#include <cstdint>  // std::size_t

#include "is_same.h"

namespace kk {
template <typename T>
struct remove_all_extents {
  using type = T;
};

template <typename T>
struct remove_all_extents<T[]> {
  using type = typename remove_all_extents<T>::type;
};

template <typename T, std::size_t N>
struct remove_all_extents<T[N]> {
  using type = typename remove_all_extents<T>::type;
};

template <typename T>
using remove_all_extents_t = typename remove_all_extents<T>::type;

static_assert(is_same_v<int, remove_all_extents_t<int>>);
static_assert(is_same_v<int, remove_all_extents_t<int[]>>);
static_assert(is_same_v<int, remove_all_extents_t<int[][3]>>);
static_assert(is_same_v<int, remove_all_extents_t<int[2][3][4][5]>>);

// CAUTION
static_assert(
    is_same_v<const char (&)[3], remove_all_extents_t<decltype("ab")>>);
}  // namespace kk
