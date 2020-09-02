#include <stdarg.h>
#include <stdio.h>
#include <unistd.h> // STDOUT_FILENO

void test_fprintf() { fprintf(stdout, "hello %s\n", "fprintf"); }

void test_dprintf() { dprintf(STDOUT_FILENO, "hello %s\n", "dprintf"); }

void test_sprintf() {
  char buf[100];
  sprintf(buf, "hello %s\n", "sprintf");
  printf("%s", buf);
}

void test_snprintf() {
  char buf[9];
  snprintf(buf, sizeof(buf), "hello %s\n", "snprintf");
  printf("%s\n", buf); // hello s
}

void test_vprintf(const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  vprintf(format, ap);
  va_end(ap);
}

void test_printf() {
  printf("o: %#o\n", 15);   // 017
  printf("x: %#x\n", 15);   // 0xf
  printf("X: %#X\n", 15);   // 0XF
  printf("a: %#a\n", 1.25); // 0x1.4p+0
  printf("A: %#A\n", 1.25); // 0X1.4P+0
  printf("e: %#e\n", 1.25); // 1.250000e+00
  printf("E: %#E\n", 1.25); // 1.250000E+00
  printf("f: %#f\n", 1.25); // 1.250000
  printf("F: %#F\n", 1.25); // 1.250000
  printf("g: %#g\n", 1.25); // 1.25000
  printf("G: %#G\n", 1.25); // 1.25000

  printf("05d: %05d\n", 15); // 05d: 00015
  printf(" 5d: % 5d\n", 15); //  5d:    15
}

int main() {
  test_vprintf("hello %s\n", "vprintf");
  test_fprintf();
  test_dprintf();
  test_sprintf();
  test_snprintf();
  test_printf();
  return 0;
}
