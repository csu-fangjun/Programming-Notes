#include <cstdarg>
#include <iostream>
#include <string>

// refer to `man stdarg`

int sum(int n, ...) {
  va_list va;
  va_start(va, n);
  int s = 0;
  for (int i = 0; i != n; ++i) {
    int p = va_arg(va, int);
    s += p;
  }
  va_end(va);
  return s;
}

int product(int n, va_list va) {
  int s = 1;
  for (int i = 0; i != n; ++i) {
    int p = va_arg(va, int);
    s *= p;
  }
  return s;
}

int product(int n, ...) {
  va_list va;
  va_start(va, n);
  int s = product(n, va);
  va_end(va);
  return s;
}

int test_vprintf(const char *format, va_list va) { vprintf(format, va); }
int test_vprintf(const char *format, ...) {
  va_list va;
  va_start(va, format);
  int i = test_vprintf(format, va);
  va_end(va);
}

std::string printf_to_string(const char *format, va_list va) {
  std::string s;

  va_list backup;
  va_copy(backup, va);

  char buf[2]; // char buf[1024];
  int result = vsnprintf(buf, sizeof(buf), format, backup);
  va_end(backup);

  if (result > 0 && result < sizeof(buf)) {
    s.append(buf, result);
    return s;
  }

  int length = sizeof(buf);
  while (true) {
    if (result < 0) {
      length *= 2;
    } else {
      length += 1;
    }
    char *buffer = new char[length];
    va_copy(backup, va);
    result = vsnprintf(buffer, length, format, backup);
    va_end(backup);

    if (result > 0 && result < length) {
      s.append(buffer, result);
      delete[] buffer;
      return s;
    }

    delete[] buffer;
  }
}

std::string string_printf(const char *format, ...) {
  va_list va;
  va_start(va, format);
  std::string s = printf_to_string(format, va);
  va_end(va);
  return s;
}

int main() {
  std::cout << "sum(1, 2, 3): " << sum(3, 1, 2, 3) << "\n";
  std::cout << "product(1, 2, 3): " << product(3, 1, 2, 3) << "\n";
  test_vprintf("%s %d\n", "hello", 123);

  auto s = string_printf("hello %s\n", "world");
  std::cout << s;
  return 0;
}
