#include <iostream>

struct swallow {
  template <typename... Args>
  swallow(Args&&...) {}
};

struct unit {};

template <typename... Args>
void print(Args&&... args) {
  swallow{std::cout << args << " " ...};
}

template <typename T>
T add(T t) {
  return t;
}

// template <typename T, typename... Args>
template <typename T, typename... Args>
T add(T t, Args... tail) {
  return t + add(tail...);
}

template <typename T, T... tail>
T add(T t) {
  return t + add(tail...);
}

struct Hole {
  template <typename... Args>
  Hole(Args&&...) {}
};
template <typename... Args>
void print2(Args&&... args) {
  Hole h{(std::cout << args << " ", 0)...};
}

template <int... i>
struct Adder;

template <>
struct Adder<> {
  enum { value = 0 };
};
template <int i, int... s>
struct Adder<i, s...> {
  enum { value = i + Adder<s...>::value };
};

template <typename... Args>
static void print3(Args&&... args) {
  int a[] = {(std::cout << args << ", ", 0)...};
  (void)a;
}

int main() {
  // print(1, 2);
  // print(add(1, 2, 3, 4, 5));
  // print(add<int, 1, 2, 3, 4>(5));
  // print2(10, 20, 30);
  // print(Adder<1, 2, 3, 4, 5>::value);
  print3(100, 2000);
}
