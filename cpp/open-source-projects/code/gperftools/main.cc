#include <stdlib.h>
#include <unistd.h>

void *test() {
  void *p = malloc(1 << 20);
  return p;
}

int main() {
  void *p = test();
  p = test();

  return 0;
}
