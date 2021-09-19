#include "torch/torch.h"
#include <cassert>
#include <iostream>

// see c10/util/ArrayRef.h

// ArrayRef<T> contains a const data pointer and a size_t.
//
// it supports `std::cout<<`
//
// it is very cheap to copy an ArrayRef, so it is usually passed
// by value
//
// using IntArrayRef == ArrayRef<int64_t>.
// Note that it is `int64_t`.
//
// It has many methods that std::vector<t> has
// It also support slicing
//
// We can construct an ArrayRef<T> from
//  (1) an initializer list
//  (2) a std::vector
//  (3) a range identified by two pointers: begin, end
//  (4) a range identified by a pointer and a length
//  (5) a std::array
//  (6) a c array
//
// It has a method `vec()` that convers *this to a std::vector by copying
//
// It supports assignment, by not move assignment.
//
// It has a default constructor
//
// It supports comparing between two ArrayRef
// or between a std::vector and an ArrayRef.
//

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
    // from an initializer
    torch::IntArrayRef arr = {1, 2, 3, 4};
    assert(arr.size() == 4);
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);
    assert(arr[3] == 4);

    // remove the first one element
    // arr[1:]
    assert((arr.slice(1) == torch::IntArrayRef({2, 3, 4})));

    // remove the first two elements
    // arr[2:]
    assert((arr.slice(2) == torch::IntArrayRef({3, 4})));

    // arr[1:(1+2)]
    // start from 1 and take 2 elements
    assert((arr.slice(1, 2) == torch::IntArrayRef({2, 3})));

    assert(arr.front() == 1);
    assert(arr.back() == 4);
    assert(arr.empty() == false);

    torch::IntArrayRef arr2 = arr;
    assert(arr == arr2);
  }
}

} // namespace

void test_array_ref() { test(); }
