//  use after free
//
// gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex1.c -o ex1
//
#include <stdlib.h>

int main() {
  char *x = (char *)malloc(10 * sizeof(char));
  free(x);
  return x[5];
}
