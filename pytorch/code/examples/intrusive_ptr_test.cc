#include <cassert>
#include <iostream>

#include "c10/util/intrusive_ptr.h"

namespace {
// c10::intrusive_ptr_target contains
//  std::atomic<size_t> refcount;
//  std::atomic<size_t> weakcount;
//  and a virtual table pointer.
static_assert(sizeof(std::atomic<size_t>) == 8, "");
static_assert(
    sizeof(c10::intrusive_ptr_target) == 16 + 8,
    "It should contain only two std::atomic<size_t> members plus a vptr");
class Foo : public c10::intrusive_ptr_target {
public:
  Foo(int i) : a_(i) {}
  int GetA() const { return a_; }
  void SetA(int a) { a_ = a; }

private:
  int a_;
};

// +8 because there are 4-bytes for padding
static_assert(sizeof(Foo) == sizeof(c10::intrusive_ptr_target) + 8);

void test_intrusive_ptr_impl() {
  {
      // compile time error since this constructor is private
      // c10::intrusive_ptr<Foo> foo(new Foo(10));
  }

  // Note that there is only one way to construct
  // an c10::intrusive_ptr
  {
    // it calls `new Foo(10)` internally in c10::make_intrusive
    c10::intrusive_ptr<Foo> foo = c10::make_intrusive<Foo>(10);
    static_assert(sizeof(c10::intrusive_ptr<Foo>) == 8,
                  "it should contain only a single pointer.");
    assert(foo->GetA() == 10);
    assert((*foo).GetA() == 10);

    assert(foo.unique() == true);      // refcount == 1
    assert(foo.use_count() == 1);      // refcount
    assert(foo.weak_use_count() == 1); // weakcount
    assert(!foo == false);
    assert(foo.defined() == true); // refcount > 0

    // usually we should not call release directly
    // if we indeed need to call release, remember to
    // use reclaim()
    Foo *p = foo.release();

    // now foo is empty
    assert(foo.use_count() == 0);
    assert((bool)foo == false);
    assert(!foo == true);
    assert(foo.use_count() == 0);
    assert(foo.weak_use_count() == 0);
    assert(foo.defined() == false);

    foo = c10::intrusive_ptr<Foo>::reclaim(p);
    // now p is owned by foo
    assert(foo.get() == p);
    assert(foo.defined() == true);

    foo.reset();
    assert(foo.get() == nullptr);
    assert(foo.defined() == false);
  }
  {
    // copy and assign
    auto foo = c10::make_intrusive<Foo>(100);

    auto bar = foo; // copy constructor
    assert(bar.use_count() == 2);
    assert(foo.use_count() == 2);

    bar.reset();
    assert(foo.use_count() == 1);

    c10::intrusive_ptr<Foo> baz = std::move(foo); // move copy constructor
    assert(baz.use_count() == 1);
    assert(foo.use_count() == 0);

    foo = std::move(baz); // move assignment
    assert(foo.use_count() == 1);
    assert(baz.use_count() == 0);
  }
}

void test_weak_intrusive_ptr() {
  {
    auto foo = c10::make_intrusive<Foo>(10);

    // the first weak pointer can only be constructed
    // for a strong pointer, i.e., c10::intrusive_ptr
    c10::weak_intrusive_ptr<Foo> bar(foo);
    assert(foo.use_count() == 1);
    assert(foo.weak_use_count() == 2);

    assert(bar.use_count() == 1);
    assert(bar.weak_use_count() == 2);

    // we have to use lock() to get a strong pointer
    auto baz = bar.lock();
    assert(baz.defined() == true);
    assert(baz.use_count() == 2); // both foo and baz are strong
    baz.reset();

    assert(bar.expired() == false);
    foo.reset();

    assert(bar.use_count() == 0);
    assert(bar.weak_use_count() == 1);
    assert(bar.expired() == true);

    baz = bar.lock(); // since bar is expired(), lock() returns a nullptr
    assert(baz.defined() == false);
  }
}

} // namespace

void test_intrusive_ptr() {
  test_intrusive_ptr_impl();
  test_weak_intrusive_ptr();
}
