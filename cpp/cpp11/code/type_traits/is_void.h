#pragma once

#include "is_same.h"
#include "remove_cv.h"

namespace kk {

// CAUTION: We have to use remove_cv here!!
// Otherwise, we have to write a lot of specializations!
template <typename T>
struct is_void : is_same<void, remove_cv_t<T>> {};

template <typename T>
inline constexpr bool is_void_v = is_void<T>::value;

static_assert(is_void_v<void> == true);
static_assert(is_void_v<const void> == true);
static_assert(is_void_v<const volatile void> == true);
static_assert(is_void_v<volatile const void> == true);
static_assert(is_void_v<void *> == false);

}  // namespace kk
