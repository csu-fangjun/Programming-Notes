#include <iostream>

// refer to https://burtleburtle.net/bob/c/lookup3.c
// clang-format off

#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

#define mix(a,b,c) \
{ \
    a -= c;  a ^= rot(c, 4);  c += b; \
    b -= a;  b ^= rot(a, 6);  a += c; \
    c -= b;  c ^= rot(b, 8);  b += a; \
    a -= c;  a ^= rot(c,16);  c += b; \
    b -= a;  b ^= rot(a,19);  a += c; \
    c -= b;  c ^= rot(b, 4);  b += a; \
}
// clang-format on

int main() {
  int a = 1;
  int b = 2;
  int c = 3;

  mix(a, b, c);
  std::cout << "a: " << a << "\n";
  std::cout << "b: " << b << "\n";
  std::cout << "c: " << c << "\n";
  return 0;
}
