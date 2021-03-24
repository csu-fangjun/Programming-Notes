#pragma once

#include <utility>  // for std::declval

#include "integral_constant.h"
#include "void_t.h"

namespace kk {

template <typename Lhs, typename Rhs, typename = void>
struct is_assignable_helper : false_type {};

template <typename Lhs, typename Rhs>
struct is_assignable_helper<
    Lhs, Rhs, void_t<decltype(std::declval<Lhs>() = std::declval<Rhs>())>>
    : true_type {};

template <typename Lhs, typename Rhs>
struct is_assignable : is_assignable_helper<Lhs, Rhs> {};

template <typename Lhs, typename Rhs>
inline constexpr bool is_assignable_v = is_assignable<Lhs, Rhs>::value;

static_assert(is_assignable_v<int, int> == false);  // CAUTION
static_assert(is_assignable_v<int &, int>);
static_assert(is_assignable_v<int &, const int>);
static_assert(is_assignable_v<int &&, int> == false);

static_assert(is_assignable_v<int &, double>);
static_assert(is_assignable_v<int *&, int *>);
static_assert(is_assignable_v<int *&, int *const>);
static_assert(is_assignable_v<int *&, const int *> == false);

}  // namespace kk
