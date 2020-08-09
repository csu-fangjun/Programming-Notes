
// gcc -S -o add_no_cfi.s -fno-asynchronous-unwind-tables add.c
// gcc -S add.c

int add(int a, int b) {
  int c = a + b;
  return c;
}
