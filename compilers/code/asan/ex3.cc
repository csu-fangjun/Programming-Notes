// malloc + delete
// gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex3.cc -o ex3
#include <stdlib.h>

int main() {
  int *p = (int *)malloc(sizeof(int));
  delete p;
  return 0;
}
