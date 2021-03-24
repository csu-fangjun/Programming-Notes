#pragma once

#include "integral_constant.h"
#include "is_same.h"

namespace kk {

template <typename T>
struct is_nullptr : false_type {};

template <>
struct is_nullptr<decltype(nullptr)> : true_type {};

// template <>
// struct is_nullptr<decltype(nullptr)> : true_type {};

template <typename T>
inline constexpr bool is_nullptr_v = is_nullptr<T>::value;

static_assert(is_nullptr_v<decltype(nullptr)>);

}  // namespace kk
