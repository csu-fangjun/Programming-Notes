// new + out of bound write
// gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex9.cc -o ex9
#include <memory>

int main() {
  auto p = new int[1];

  int a = p[1]; // this one cannot be detected!
  p[2] = 3;

  delete[] p;
  return 0;
}
