

#define TEST(x)                                                                \
  void test_##x();                                                             \
  test_##x();

int main() {
  // TEST(hello);
  TEST(tensor);
  return 0;
}
