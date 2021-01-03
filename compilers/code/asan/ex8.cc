// new + memory leak
// gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex8.cc -o ex8
#include <memory>

int main() {
  auto p = new int;
  return 0;
}
