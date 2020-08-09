
// man va_arg
#include <stdarg.h>
#include <stdio.h>

static int sum(int n, ...) {
  va_list ap;
  // every call of va_start must have a corresponding va_end
  va_start(ap, n);
  int s = 0;
  for (int i = 0; i != n; ++i) {
    int v = va_arg(ap, int);
    s += v;
  }
  va_end(ap);
  return s;
}

static int sum2(int n, va_list ap) {
  int s = 0;
  for (int i = 0; i != n; ++i) {
    int v = va_arg(ap, int);
    s += v;
  }
  return s;
}

static int adder(int n, ...) {
  va_list ap;
  // every call of va_start must have a corresponding va_end
  va_start(ap, n);
  int s = sum2(n, ap);
  va_end(ap);
  return s;
}

void my_printf(const char* msg, const char* format, ...) {
  fflush(stdout);  // flush any pending output
  printf("hello: %s\n", msg);
  va_list ap;
  va_start(ap, format);
  vprintf(format, ap);
  va_end(ap);
}

int main() {
  printf("sum(3, 10, 20, 30) = %d\n", sum(3, 10, 20, 30));
  printf("sum(1, 10) = %d\n", sum(1, 10));
  printf("adder(3, 10, 20, 30) = %d\n", adder(3, 10, 20, 30));
  printf("adder(1, 10) = %d\n", adder(1, 10));

  my_printf("world", "%s %d\n", "here", 10);
  return 0;
}
