// new + free
// gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex4.cc -o ex4
#include <stdlib.h>

int main() {
  int *p = new int;
  free(p);
  return 0;
}
