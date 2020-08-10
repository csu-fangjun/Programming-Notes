#include <stdio.h>
#include <stdlib.h>

int main() {
  int *a = (int *)malloc(sizeof(int));
  printf("a is %d, %p\n", *a, a);

  *a = 100;
  int *b = (int *)malloc(sizeof(int));
  printf("b is %d, %p\n", *b, b);

  free(b);
  free(a);

  return 0;
}
