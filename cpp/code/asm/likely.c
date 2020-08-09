// gcc -S -fno-asynchronous-unwind-tables likely.c
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

int test_likely(int a) { return a + 1; }
int test_unlikely(int a) { return a + 2; }

int test0(int a) {
  if (a) {
    return test_likely(a);
  } else {
    return test_unlikely(a);
  }
}

int test1(int a) {
  if (likely(a)) {
    return test_likely(a);
  } else {
    return test_unlikely(a);
  }
}

int test2(int a) {
  if (unlikely(a)) {
    return test_unlikely(a);
  } else {
    return test_likely(a);
  }
}
