#pragma once
#include "integral_constant.h"

namespace kk {
template <typename T>
struct is_lvalue_reference : false_type {};

template <typename T>
struct is_lvalue_reference<T &> : true_type {};

// CAUTION
template <typename T>
inline constexpr bool is_lvalue_reference_v = is_lvalue_reference<T>::value;

static_assert(is_lvalue_reference_v<int> == false);
static_assert(is_lvalue_reference_v<int &> == true);
static_assert(is_lvalue_reference_v<int &&> == false);

static_assert(is_lvalue_reference_v<int (&)[]> == true);
static_assert(is_lvalue_reference_v<int(&&)[]> == false);

static_assert(is_lvalue_reference_v<decltype("hello")> == true);

}  // namespace kk
