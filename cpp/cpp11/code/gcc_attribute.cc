#include <cstdarg>
#include <cstdio>
// refer to
//
// abseil-cpp/absl/base/attributes.h
//
// https://isocpp.org/std/standing-documents/sd-6-sg10-feature-test-recommendations

// ============================================================
//  __has_attribute
// ------------------------------------------------------------
//
#ifdef __has_attribute
// refer to
// https://gcc.gnu.org/onlinedocs/cpp/_005f_005fhas_005fattribute.html
// and
// https://clang.llvm.org/docs/LanguageExtensions.html#has-attribute
#define KK_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#define KK_HAS_ATTRIBUTE(x) (0)
#endif

// ============================================================
//  __has_cpp_attribute
// ------------------------------------------------------------
//
// refer to
// https://gcc.gnu.org/onlinedocs/cpp/_005f_005fhas_005fcpp_005fattribute.html
// and
// https://clang.llvm.org/docs/LanguageExtensions.html#has-cpp-attribute
#if defined(__cplusplus) && defined(__has_cpp_attribute)
#define KK_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define KK_HAS_CPP_ATTRIBUTE(x) (0)
#endif

// ============================================================
//  fall through
// ------------------------------------------------------------
//
#if defined(__GNUC__) && __GNUC__ > 7
#define KK_FALL_THROUGH [[gnu::fallthrough]]
#elif KK_HAS_ATTRIBUTE(fallthrough)
#define KK_FALL_THROUGH __attribute__((fallthrough))
#elif defined(__cplusplus) && (__cplusplus >= 201703L)
// refer to https://en.cppreference.com/w/cpp/language/attributes/fallthrough
#define KK_FALL_THROUGH [[fallthrough]]
#else
// a no-op;
// note that it still results in a warning when -Wimplicit-fallthrough is used
#define KK_FALL_THROUGH                                                        \
  do {                                                                         \
  } while (0)
#endif

// ============================================================
//  function attribute
// ------------------------------------------------------------
//
// https://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html

// ============================================================
//  format attribute
// ------------------------------------------------------------
//
#if KK_HAS_ATTRIBUTE(format)
#define KK_PRINTF_ATTRIBUTE(string_index, first_to_check)                      \
  __attribute__((format(printf, string_index, first_to_check)))

#else
#define KK_PRINTF_ATTRIBUTE(string_index, first_to_check)
#endif

// ============================================================
//  unused
// ------------------------------------------------------------
//  -Wunused
//
#if KK_HAS_ATTRIBUTE(unused)
#define KK_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define KK_ATTRIBUTE_UNUSED
#endif

// ============================================================
//  deprecated
// ------------------------------------------------------------
//
#if KK_HAS_ATTRIBUTE(deprecated)
#define KK_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#define KK_DEPRECATED(msg)
#endif

int test_fall_through(int n) {
  int a = 1;
  switch (n) {
  case 10:
    a = 3; // fall through
  case 11:
    a = 4;
    KK_FALL_THROUGH;
  case 12:
    a = 12; // there would be a warning
  default:
    a = 0;
  }
  return a;
}

// compile it with -Wformat
//
// format is the second argument, so we use 2
// the paremeters for format starts from 3, so we use 3
//
// Note that it counts from 1
KK_PRINTF_ATTRIBUTE(2, 3)
void test_format_attribute(int i, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  vprintf(format, ap);
  va_end(ap);
}

class TestFormat {
public:
  // note that it starts from
  // the first parameter is this
  // format is 3
  // the argument for format is 4
  KK_PRINTF_ATTRIBUTE(3, 4)
  void test_format_attribute(int i, const char *format, ...) {
    va_list ap;
    va_start(ap, format);
    vprintf(format, ap);
    va_end(ap);
  }
};

int main() {
  test_fall_through(10);
  test_format_attribute(10, "%s, %s\n", "hello"); // g++ -Wformat xxx.cc
                                                  // emits a warning
  TestFormat f;
  f.test_format_attribute(10, "%s, %s\n", "hello"); // warning for -Wformat
  int ab;                                           // a warning for unused
  int bc KK_ATTRIBUTE_UNUSED;                       // no warning for unused

  int d KK_DEPRECATED("do not used d!");
  d = 10;  // warning, -Wdeprecated-declarations
  (void)d; // warning, -Wdeprecated-declarations
}
