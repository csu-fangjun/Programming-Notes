
#include "torch/torch.h"
#include <stdlib.h>

static void test_inefficient_std_function_context() {
  int *p = new int{3};
  auto deleter = [](void *q) {
    // Note: q is the same as p
    std::cout << "q is: " << *(int *)q << "\n";
    delete (int *)q;
  };

  // the pointer inside the returned ptr is `p`,
  // the context is an instance of torch::InefficientStdFunctionContext, which
  // contains the given `deleter` and it uses the `deleter` to delete `p`.
  torch::DataPtr ptr = torch::InefficientStdFunctionContext::makeDataPtr(
      p, deleter, torch::kCPU);
  // When ptr is freed, the following happens:
  //  (1) UniqueVoidPtr. It deletes the passed InefficientStdFunctionContext
  //  (2) InefficientStdFunctionContext contains a unique ptr, who is managing
  //  `p` and its deleter is the passed `deleter`, i.e., the lambda defined
  //  above.

  {
    auto deleter = [](void *q) {
      std::cout << "free q: " << intptr_t(q) << "\n";
      free(q);
    };

    void *p = malloc(10);
    std::cout << "p is: " << intptr_t(p) << "\n";
    torch::DataPtr ptr = torch::InefficientStdFunctionContext::makeDataPtr(
        p, deleter, torch::kCPU);
    // We have a pointer and a deleter. The deleter is used to free the pointer
  }
}

static void my_free(void *q) {
  // used in `test_data_ptr()`
  std::cout << "free " << intptr_t(q) << "\n";
  free(q);
}

static void test_data_ptr() {
  // see c10/core/Allocator.h
  //
  {
    torch::DataPtr p; // default constructor
    assert(p.get() == nullptr);
    assert(p.get_context() == nullptr);
    assert(!p == true);
    assert(p == false);
    assert(p.get_deleter() ==
           &c10::detail::deleteNothing); // see c10/util/UniqueVoidPtr.h

    // Note: the type is torch::Device, but we use
    // torch::DeviceType, i.e., torch::kCPU here
    assert(p.device() == torch::kCPU);
  }

  {
    int32_t a[3];
    // the pointer in torch::DataPtr is borrowed, so we don't pass
    // a context to it.
    torch::DataPtr p(a, torch::kCPU);
    assert(p.get() == a);
    assert(p.get_context() == nullptr);
    assert(!p == false);
    assert(p == true);
    assert(p.get_deleter() == c10::detail::deleteNothing);
  }

  {
    int32_t *q = reinterpret_cast<int32_t *>(malloc(3 * sizeof(int32_t)));

    // the memory is allocated by us, so we need a deleter
    // here the context is the same as the pointer. The deleter
    // will delete the context
    // torch::DataPtr p(q, q, &free, torch::kCPU);
    std::cout << "q is: " << intptr_t(q) << "\n";
    torch::DataPtr p(q, q, &my_free, torch::kCPU);
    assert(p.get_deleter() == &my_free);
  }

  {
    char *q = reinterpret_cast<char *>(malloc(50 * sizeof(char)));
    // double *r = reinterpret_cast<double *>((intptr_t(q) + 31) & ~31);
    double *r = reinterpret_cast<double *>((intptr_t(q) + 31) & ~31) + 1;
    std::cout << "q: " << intptr_t(q) << ", " << intptr_t(q) % 32 << "\n";
    std::cout << "r: " << intptr_t(r) << ", " << intptr_t(r) % 32 << "\n";
    torch::DataPtr p(r, q, &my_free, torch::kCPU);
    // Note: Now the pointer and the context are not the same.
    // The deleter will free `q`, not `r`.
    assert(p.get() == r);
    assert(p.get_context() == q);
    assert(p.get_deleter() == &my_free);
    p.clear();
    assert(p.get() == nullptr);
  }
}

// in namespace c10, defined in c10/util/UniqueVoidPtr
//
// using DeleterFnPtr = void (*)(void*);
namespace {
class MyAllocator : public torch::Allocator {
public:
  static void Deleter(void *p) {
    std::cout << "free " << intptr_t(p) << "\n";
    free(p);
  }
  // raw_deleter returns a non-null ptr, it means:
  //  - the deleter in the return value of `allocate()` is the same
  //    as the return value of `raw_deleter`
  //
  //  - We can use `raw_deallocate()` to free the pointer returned by
  //  `raw_allocate()`
  torch::DeleterFnPtr raw_deleter() const override { return &Deleter; }
  torch::DataPtr allocate(size_t n) const override {
    if (n == 0u) {
      std::cout << "return a null ptr\n";
      torch::DataPtr ans(/*data*/ nullptr, /*ctx*/ nullptr,
                         /*DeleterFnPtr*/ nullptr, /*device*/ torch::kCPU);
      return ans;
    }
    void *p = malloc(n);
    torch::DataPtr ans(/*data*/ p, /*ctx*/ p,
                       /*DeleterFnPtr*/ &Deleter, /*device*/ torch::kCPU);
    std::cout << "return " << intptr_t(p) << "\n";
    return ans;
  }
};
} // namespace

static void test_my_allocator() {
  MyAllocator allocator;
  torch::DataPtr p = allocator.allocate(2 * sizeof(int32_t));
  int32_t *q = reinterpret_cast<int32_t *>(p.get());
  q[0] = 10;
  q[1] = 20;

  // since MyAllocator::raw_deleter() returns a non-null pointer,
  // we can use `raw_allocate` to allocate memory and use `raw_deallocate()`
  // to free the memory.
  q = reinterpret_cast<int32_t *>(allocator.raw_allocate(1 * sizeof(int32_t)));
  q[0] = 20;
  allocator.raw_deallocate(q);
}

static void test() {
  test_my_allocator();
  test_data_ptr();
  test_inefficient_std_function_context();
}

void test_allocator() { test(); }
