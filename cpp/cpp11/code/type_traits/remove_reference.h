#pragma once

#include "is_same.h"

namespace kk {
template <typename T>
struct remove_reference {
  using type = T;
};

template <typename T>
struct remove_reference<T &> {
  using type = T;
};

template <typename T>
struct remove_reference<T &&> {
  using type = T;
};

template <typename T>
using remove_reference_t = typename remove_reference<T>::type;

static_assert(is_same_v<int, remove_reference_t<int &>>);
static_assert(is_same_v<int, remove_reference_t<int &&>>);
static_assert(is_same_v<volatile int, remove_reference_t<volatile int &>>);

}  // namespace kk
