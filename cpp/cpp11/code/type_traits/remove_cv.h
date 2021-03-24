#pragma once
#include "is_same.h"
#include "remove_const.h"
#include "remove_volatile.h"

namespace kk {
#if 0
template <typename T>
struct remove_cv {
  using type = T;
};

template <typename T>
struct remove_cv<const T> {
  using type = T;
};

template <typename T>
struct remove_cv<volatile T> {
  using type = T;
};

template <typename T>
struct remove_cv<const volatile T> {
  using type = T;
};
#else
template <typename T>
struct remove_cv {
  using type = remove_const_t<remove_volatile_t<T>>;
};
#endif

template <typename T>
using remove_cv_t = typename remove_cv<T>::type;

static_assert(is_same_v<int, remove_cv_t<const int>>);
static_assert(is_same_v<int, remove_cv_t<volatile int>>);

// CAUTION
static_assert(is_same_v<volatile int *, remove_cv_t<volatile int *>>);
static_assert(
    is_same_v<volatile const int *, remove_cv_t<const volatile int *>>);

static_assert(is_same_v<volatile int *, remove_cv_t<volatile int *volatile>>);

}  // namespace kk
