// new[] + delete
// gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex5.cc -o ex5
#include <stdlib.h>

int main() {
  int *p = new int[2];
  delete p;
  return 0;
}
