

#define TEST(x)                                                                \
  void Test##x();                                                              \
  Test##x();

int main() {
  // TEST(hello);
  // TEST(tensor);
  // TEST(cuda);
  // TEST(intrusive_ptr);
  TEST(Device);
  return 0;
}
