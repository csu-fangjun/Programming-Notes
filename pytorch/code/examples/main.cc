

#define TEST(x)                                                                \
  void test_##x();                                                             \
  test_##x();

int main() {
  TEST(array_ref)
  TEST(cuda);
  TEST(device);
  TEST(hello);
  TEST(intrusive_ptr);
  TEST(scalar_type)
  TEST(tensor);
  return 0;
}
