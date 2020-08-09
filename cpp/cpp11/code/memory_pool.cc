#include <cassert>
#include <iostream>
#include <memory>
#include <new>

namespace k1 {

class MemoryArena {
 public:
  MemoryArena(size_t capacity_in_bytes) {
    char* p = new char[capacity_in_bytes];
    base_ = p;
    end_ = p + capacity_in_bytes;
    buf_.reset(p);
  }

  void* Allocate(size_t bytes) {
    assert(base_ + bytes < end_);
    char* p = base_;
    base_ += bytes;
    return base_;
  }

  void Destroy(void* p) { std::cout << "Destroying " << p << "\n"; }

 private:
  std::unique_ptr<char[]> buf_;
  char* base_;
  char* end_;
};

class Integer {
 public:
  ~Integer() { std::cout << "destructor " << this << " is called\n"; }
  static void* operator new(size_t bytes, MemoryArena& arena) {
    char* p =
        reinterpret_cast<char*>(arena.Allocate(bytes + sizeof(MemoryArena*)));
    auto** pa = reinterpret_cast<MemoryArena**>(p);
    *pa = &arena;

    p += sizeof(MemoryArena*);

    return p;
  }
  void operator delete(void* p, MemoryArena& arena) {
    // this function is called only when the constructor throws an exception.
    // refer to https://en.cppreference.com/w/cpp/memory/new/operator_delete
    // item (15)
  }
  void operator delete(void* _p) {
    char* p = reinterpret_cast<char*>(_p);
    MemoryArena* pa = reinterpret_cast<MemoryArena*>(p - sizeof(MemoryArena*));
    pa->Destroy(pa);
  }

 private:
  int i_;
};
void test() {
  MemoryArena arena(100);
  Integer* i = new (arena) Integer;
  std::cout << "value of i: " << i << "\n";

  Integer* p = new (arena) Integer;
  std::cout << "value of p: " << p << "\n";

  delete i;
  delete p;
}

}  // namespace k1

namespace k2 {
template <typename T>
class MemoryArena {
 public:
 private:
  int block_size_;
  char* base_ = nullptr;
  std::list<std::unique_ptr<char[]>> buf_;
};
}  // namespace k2

int main() {
  k1::test();
  return 0;
}
