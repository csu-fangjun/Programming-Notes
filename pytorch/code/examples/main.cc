

#define TEST(x)                                                                \
  void test_##x();                                                             \
  test_##x();

int main() {
  // TEST(hello);
  // TEST(tensor);
  TEST(cuda);
  // TEST(intrusive_ptr);
  return 0;
}
