// new + delete[]
// gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex6.cc -o ex6
#include <stdlib.h>

int main() {
  int *p = new int;
  delete[] p;
  return 0;
}
