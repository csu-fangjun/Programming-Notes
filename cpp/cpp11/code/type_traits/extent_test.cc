
template <typename T>
struct Extent {};

template <typename T>
struct Extent<T[]> : Extent<T> {};

template <typename T, int N>
struct Extent<T[N]> : Extent<T> {};

int main() {
  Extent<int> f1;
  Extent<int[]> f2;
  Extent<int[1]> f3;
  Extent<int[][2]> f4;
  Extent<int[][3][4]> f5;
}

// The following the the output from https://cppinsights.io/

template <typename T>
struct Extent {};

/* First instantiated from: insights.cpp:13 */
#ifdef INSIGHTS_USE_TEMPLATE
template <>
struct Extent<int> {
  // inline constexpr Extent() noexcept = default;
};

#endif

/* First instantiated from: insights.cpp:14 */
#ifdef INSIGHTS_USE_TEMPLATE
template <>
struct Extent<int[]> : public Extent<int> {
  // inline constexpr Extent() noexcept = default;
};

#endif

/* First instantiated from: insights.cpp:15 */
#ifdef INSIGHTS_USE_TEMPLATE
template <>
struct Extent<int[1]> : public Extent<int> {
  // inline constexpr Extent() noexcept = default;
};

#endif

/* First instantiated from: insights.cpp:16 */
#ifdef INSIGHTS_USE_TEMPLATE
template <>
struct Extent<int[][2]> : public Extent<int[2]> {
  // inline constexpr Extent() noexcept = default;
};

#endif

/* First instantiated from: insights.cpp:6 */
#ifdef INSIGHTS_USE_TEMPLATE
template <>
struct Extent<int[2]> : public Extent<int> {
  // inline constexpr Extent() noexcept = default;
};

#endif

/* First instantiated from: insights.cpp:17 */
#ifdef INSIGHTS_USE_TEMPLATE
template <>
struct Extent<int[][3][4]> : public Extent<int[3][4]> {
  // inline constexpr Extent() noexcept = default;
};

#endif

/* First instantiated from: insights.cpp:6 */
#ifdef INSIGHTS_USE_TEMPLATE
template <>
struct Extent<int[3][4]> : public Extent<int[4]> {
  // inline constexpr Extent() noexcept = default;
};

#endif

/* First instantiated from: insights.cpp:9 */
#ifdef INSIGHTS_USE_TEMPLATE
template <>
struct Extent<int[4]> : public Extent<int> {
  // inline constexpr Extent() noexcept = default;
};

#endif

template <typename T>
struct Extent<T[]> : public Extent<T> {};

template <typename T, int N>
struct Extent<T[N]> : public Extent<T> {};

int main() {
  Extent<int> f1 = Extent<int>();
  Extent<int[]> f2 = Extent<int[]>();
  Extent<int[1]> f3 = Extent<int[1]>();
  Extent<int[][2]> f4 = Extent<int[][2]>();
  Extent<int[][3][4]> f5 = Extent<int[][3][4]>();
}
