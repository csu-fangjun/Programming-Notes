#pragma once

namespace kk {

template <typename T, T val>
struct integral_constant {
  using value_type = T;
  // using type = integral_constant<T, val>;
  using type = integral_constant;
  static constexpr T value = val;
  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

template <bool val>
using bool_constant = integral_constant<bool, val>;

using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

static_assert(true_type::value == true, "");
static_assert(false_type::value == false, "");

static_assert(true_type{});
static_assert(false_type{} == false);

static_assert(true_type());
static_assert(false_type() == false);

static_assert(integral_constant<int, 2>() == 2);

}  // namespace kk
