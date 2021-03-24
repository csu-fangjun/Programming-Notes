#pragma once
#include "is_same.h"

namespace kk {
template <typename T>
struct remove_const {
  using type = T;
};

template <typename T>
struct remove_const<const T> {
  using type = T;
};

template <typename T>
using remove_const_t = typename remove_const<T>::type;

static_assert(is_same_v<int, remove_const_t<const int>>);
static_assert(is_same_v<void, remove_const_t<const void>>);

static_assert(is_same_v<const int *, remove_const_t<const int *>>);
static_assert(is_same_v<int *, remove_const_t<int *const>>);

// CAUTION
static_assert(is_same_v<const int *, remove_const_t<const int *const>>);

}  // namespace kk
