
#include "torch/torch.h"

namespace {
struct Foo {
  int32_t i = 0;
  int32_t k = 0;

  static int32_t copy_assigned;
  static int32_t copy_constructed;
  static int32_t move_constructed;
  static int32_t move_assigned;
  static int32_t constructed;

  Foo(int32_t i, int32_t k) : i(i), k(k) { ++constructed; }
  Foo(const Foo &other) : i(other.i), k(other.k) { ++copy_constructed; }
  Foo &operator=(const Foo &other) {
    i = other.i;
    k = other.k;
    ++copy_assigned;
    return *this;
  }

  Foo(Foo &&other) : i(other.i), k(other.k) { ++move_constructed; }
  Foo &operator=(Foo &&other) {
    i = other.i;
    k = other.k;
    ++move_assigned;
    return *this;
  }
};

int32_t Foo::constructed = 0;
int32_t Foo::copy_assigned = 0;
int32_t Foo::copy_constructed = 0;
int32_t Foo::move_constructed = 0;
int32_t Foo::move_assigned = 0;

} // namespace

static void test() {
  // It contains 1 bool, 1 int
  // Due to padding, its size is 8
  static_assert(sizeof(torch::optional<int32_t>) == 8);

  {
    // creat an empty optional
    torch::optional<int> t1;
    torch::optional<int> t2(torch::nullopt);
    assert(t1 == t2);

    assert(!t1 == true);
    assert(t1.has_value() == false);
    assert(t1 == torch::nullopt);

    assert(t1.value_or(100) == 100);

    t1 = 10;
    assert(t1.has_value() == true);
    assert(t1.value() == 10); // throw an exception if t1 is empty
    assert(*t1 == 10);        // undefined behavior if t1 is empty

    t1.reset();
    assert(t1.has_value() == false);
  }

  {
    // Note: f does not has a default constructor
    torch::optional<Foo> f;
    Foo t(10, 20); // constructed  is 1;
    f = t;         // copy constructed since f is not initialized
    assert(f->i == 10);
    assert(f->k == 20);
    assert((*f).i == 10);
    assert((*f).k == 20);
    assert(f.value().i == 10);
    assert(f.value().k == 20);

    assert(Foo::copy_constructed == 1);
    assert(Foo::copy_assigned == 0);

    f = t; // now it calls copy assignment since f is initialized
    assert(Foo::copy_constructed == 1);
    assert(Foo::copy_assigned == 1);

    f = std::move(t); // calls move assignment
    assert(Foo::move_assigned == 1);

    torch::optional<Foo> f2(t); // copy constructed
    assert(Foo::copy_constructed == 2);

    torch::optional<Foo> f3(std::move(t)); // move constructed
    assert(Foo::move_constructed == 1);
    f3 = std::move(t); // move assigned
    assert(Foo::move_assigned == 2);

    torch::optional<Foo> f4;
    f4 = std::move(t); // move constructed
    assert(Foo::move_constructed == 2);

    torch::optional<Foo> f5(torch::in_place, 100, 200); // constructor
    assert(Foo::constructed == 2);
  }
}

void test_optional() { test(); }
