// it needs torch >= 1.8.0

#include "torch/torch.h"
#define TEST(x)                                                                \
  void test_##x();                                                             \
  test_##x();

int main() {
  TEST(array_ref)
  TEST(cuda);
  TEST(device);
  TEST(device_type);
  TEST(hello);
  TEST(intrusive_ptr);
  TEST(optional);
  TEST(scalar_type)
  TEST(tensor);
  TEST(tensor_options);
  TEST(type_meta);
  TEST(allocator);
  TEST(storage_impl);
  TEST(ivalue);
  TEST(qualified_name);
  TEST(custom_class);
  return 0;
}
