
void test_inline_asm();
void test_neon_intrinsics();
void test_transpose();

extern "C" int add(int a);

int main() {
  test_inline_asm();
  test_neon_intrinsics();
  test_transpose();

  return 0;
}
