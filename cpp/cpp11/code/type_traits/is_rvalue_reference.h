#pragma once

#include "integral_constant.h"

namespace kk {
template <typename T>
struct is_rvalue_reference : false_type {};

template <typename T>
struct is_rvalue_reference<T &&> : true_type {};

template <typename T>
inline constexpr bool is_rvalue_reference_v = is_rvalue_reference<T>::value;

static_assert(is_rvalue_reference_v<int> == false);
static_assert(is_rvalue_reference_v<int &> == false);
static_assert(is_rvalue_reference_v<int &&> == true);

static_assert(is_rvalue_reference_v<int (&)[]> == false);
static_assert(is_rvalue_reference_v<int(&&)[]> == true);

// CAUTION
// decltype("ab") is of type `const char (&)[3]`
static_assert(is_rvalue_reference_v<decltype("ab")> == false);

static_assert(is_rvalue_reference_v<const char(&&)[3]> == true);

}  // namespace kk
