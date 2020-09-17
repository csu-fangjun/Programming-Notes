#include <cassert>
#include <cstdint>
#include <cstdio>

namespace {

struct Foo {
  int32_t first : 16;
  int32_t second : 16;
};

union Bar {
  struct Foo f;
  int16_t s16;
  int32_t s32;
};

void test1() {
  Bar b;
  b.s32 = 0x12345678;
  assert(b.f.first == 0x5678);
  assert(b.f.second == 0x1234);

  b.f.first = 0;
  assert(b.s16 == 0);
  assert(b.s32 == 0x12340000);
  printf("%x\n", b.s16);
  printf("%x\n", b.s32);
}

} // namespace

int main() {
  test1();
  return 0;
}
