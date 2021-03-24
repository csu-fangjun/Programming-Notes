#pragma once

#include "is_void.h"

namespace kk {

#if 0
template <typename...>
using void_t = void;

#else

template <typename...>
struct make_void {
  using type = void;
};

template <typename... Args>
using void_t = typename make_void<Args...>::type;

#endif

static_assert(is_void_v<void_t<>>);
static_assert(is_void_v<void_t<int>>);
}  // namespace kk
