// ============================================================
//  __cplusplus
// ------------------------------------------------------------
//
// See https://gcc.gnu.org/onlinedocs/cpp/Standard-Predefined-Macros.html
#if __cplusplus == 199711L // -std=c++98
#pragma message "c++98"
#elif __cplusplus == 201103L // -std=c++11, -std=c++0x
#pragma message "c++11"
#elif __cplusplus == 201402L // -std=c++14, -std=c++1y
#pragma message "c++14"
#elif __cplusplus == 201703L // -std=c++17, -std=c++1z
#pragma message "c++17"
#elif __cplusplus > 201703L
#pragma message "c++2a"
#endif

// ============================================================
//  stringify
// ------------------------------------------------------------
//
#define _TO_STR(x) #x
#define TO_STR(x) _TO_STR(x)

// ============================================================
//  Check that a macro is not a given string
// ------------------------------------------------------------
// clang-format off
// ensure that FOO is not hello
#define FOO TO_STR(world)

static_assert(FOO[0] != 'h' ||
    FOO[1] != 'e' ||
    FOO[2] != 'l' ||
    FOO[3] != 'l' ||
    FOO[4] != 'o' ||
    FOO[5] != '\0',
    "FOO cannot be hello");
// clang-format on

//============================================================
// __GNUC__
//------------------------------------------------------------
//
// Refer to https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html
//
#ifdef __GNUC__
#if __GNUC__ == 10
#pragma message "gcc major version is 10"
#endif

#if __GNUC_MINOR__ == 0
#pragma message "gcc minor version is 0"
#endif

#if __GNUC_PATCHLEVEL__ == 1
#pragma message "gcc patchlevel version is 1"
#endif

#endif

// test gcc > 10.0.0
#if __GNUC__ > 10 ||                                                           \
    (__GNUC__ == 10 &&                                                         \
     (__GNUC_MINOR__ > 0 || (__GNUC_MINOR == 0 && __GNUC_PATCHLEVEL__ > 0)))
#pragma message "GCC version is > 10.0.0"
#endif

// another approach for gcc > 9.3.2
#define GCC_VERSION                                                            \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#if GCC_VERSION > 90302
#pragma message "GCC version is > 9.3.2"
#endif

//============================================================
// byte order
//------------------------------------------------------------
#ifdef __BYTE_ORDER__
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#pragma message "little endian"
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#pragma message "big endian"
#else
#error "unsupported endian"
#endif
#endif

//============================================================
// __has_builtin
//------------------------------------------------------------
// it is available only from gcc 10+
//
// For clang, see
// http://clang.llvm.org/docs/LanguageExtensions.html#feature-checking-macros
#ifdef __has_builtin
#pragma message "__has_builtin is defined"
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

//============================================================
// has include
//------------------------------------------------------------
// for the standard header files.

#if defined(__has_include) && __has_include(<stdint.h>)
#pragma message "we can include <stdint.h>"
#endif

// not for user defined files, if foo.h is in the include path,
// it will return true; false otherwise
#if defined(__has_include) && __has_include("foo.h")
#pragma message "we can include foo.h"
#else
#pragma message "we cannot include foo.h"
#endif

int main() {}
