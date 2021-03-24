#pragma once

#include "integral_constant.h"
namespace kk {
template <typename T>
struct is_reference : false_type {};

template <typename T>
struct is_reference<T &> : true_type {};

template <typename T>
struct is_reference<T &&> : true_type {};

template <typename T>
inline constexpr bool is_reference_v = is_reference<T>::value;

static_assert(is_reference_v<int> == false);
static_assert(is_reference_v<int &> == true);
static_assert(is_reference_v<int &&> == true);
static_assert(is_reference_v<int (&)[]> == true);
static_assert(is_reference_v<int (&)()> == true);
static_assert(is_reference_v<int (*)()> == false);
static_assert(is_reference_v<int()> == false);

static_assert(is_reference_v<int(&&)[]> == true);
static_assert(is_reference_v<int(&&)()> == true);

}  // namespace kk
