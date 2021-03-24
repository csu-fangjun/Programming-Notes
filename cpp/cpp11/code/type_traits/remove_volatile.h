#pragma once
#include "is_same.h"

namespace kk {
template <typename T>
struct remove_volatile {
  using type = T;
};

template <typename T>
struct remove_volatile<volatile T> {
  using type = T;
};

template <typename T>
using remove_volatile_t = typename remove_volatile<T>::type;

static_assert(is_same_v<int, remove_volatile_t<volatile int>>);
static_assert(is_same_v<int *, remove_volatile_t<int *volatile>>);
static_assert(is_same_v<volatile int *, remove_volatile_t<volatile int *>>);

// CAUTION
static_assert(
    is_same_v<volatile int *, remove_volatile_t<volatile int *volatile>>);

}  // namespace kk
