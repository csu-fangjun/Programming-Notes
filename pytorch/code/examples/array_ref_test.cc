#include "torch/torch.h"
#include <cassert>
#include <iostream>

// ArrayRef<T> contains a const data pointer and a size_t.
//
// it supports `std::cout<<`
//
// it is very cheap to copy an ArrayRef, so it is usually passed
// by value
//
// using IntArrayRef == ArrayRef<int64_t>.
// Note that it is `int64_t`.

namespace {
void test() {
  {
    int32_t a;
    torch::ArrayRef<int32_t> arr(a);
    a = 10;
    assert(arr[0] == a);
    assert(arr.data() == &a);
    assert(arr.size() == 1);
  }

  {
    // from an array
    int32_t a[2] = {3, 8};
    torch::ArrayRef<int32_t> arr = a;
    assert(arr.size() == 2);
    assert(arr[0] == 3);
    assert(arr[1] == 8);
    assert(&arr[0] == a + 0);
    assert(&arr[1] == a + 1);
  }

  {
    // from a std::vector
    std::vector<int32_t> v{1, 2, 3};
    torch::ArrayRef<int32_t> arr(v);
    assert(arr.size() == v.size());
    assert(&arr[0] == &v[0]);
    assert(&arr[1] == &v[1]);
    assert(&arr[2] == &v[2]);
  }

  {
    // from a initializer
    torch::IntArrayRef arr = {1, 2, 3, 4};
    assert(arr.size() == 4);
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);
    assert(arr[3] == 4);
  }
}

} // namespace

void TestArrayRef() { test(); }
