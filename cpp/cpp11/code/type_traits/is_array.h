#pragma once

#include <cstdint>  // for std::size_t

#include "integral_constant.h"

namespace kk {

template <typename T>
struct is_array : false_type {};

template <typename T>
struct is_array<T[]> : true_type {};

template <typename T, std::size_t N>
struct is_array<T[N]> : true_type {};

template <typename T>
inline constexpr bool is_array_v = is_array<T>::value;

static_assert(is_array_v<int> == false);

static_assert(is_array_v<int[]>);
static_assert(is_array_v<int[2]>);

static_assert(is_array_v<int *[]>);
static_assert(is_array_v<int[][3]>);

}  // namespace kk
