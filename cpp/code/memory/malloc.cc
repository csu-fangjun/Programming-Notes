#include <assert.h>
#include <unistd.h>

#include <iostream>
#include <sstream>

// refer to
// https://danluu.com/malloc-tutorial/
namespace kk {

constexpr int32_t kMagic = 0xdeadbeaf;

struct Meta {
  struct Meta *next;
  size_t size;
  int32_t is_free;
  int32_t magic;
};

Meta *g_head = nullptr;

Meta *FindFreeBlock(Meta **last, size_t size) {
  Meta *current = g_head;
  while (current) {
    if (current->is_free && current->size >= size) {
      return current;
    }
    *last = current;
    current = current->next;
  }
  return nullptr;
}

Meta *AllocBlock(Meta *last, size_t size) {
  void *p = sbrk(0);
  void *request = sbrk(size + sizeof(Meta));
  if (request == (void *)-1) {
    // sbrk fails
    return nullptr;
  }

  assert(request == p);

  Meta *r = reinterpret_cast<Meta *>(request);
  r->next = nullptr;
  r->size = size;
  r->is_free = 0;
  r->magic = kMagic;

  if (last) {
    last->next = r;
  }

  return r;
}

void FreeBlock(Meta *p) {
  assert(p->is_free == 0);
  assert(p->magic == kMagic);
  p->is_free = 1;
}

void *malloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  Meta *last = nullptr;
  Meta *block = FindFreeBlock(&last, size);
  if (block == nullptr) {
    // no free block, allocate a new block
    block = AllocBlock(last, size);
    if (!block) {
      // failed to allocate a block
      return nullptr;
    }
  }
  block->is_free = 0;

  if (g_head == nullptr) {
    g_head = block;
  }

  void *p = block + 1;
  return p;
}

void free(void *p) {
  if (p == nullptr) {
    return;
  }

  Meta *h = reinterpret_cast<Meta *>(p) - 1;
  FreeBlock(h);
}

void PrintBlockInfo() {
  const Meta *current = g_head;
  std::ostringstream os;
  os << "\n";
  std::string sep = "";
  while (current) {
    os << sep << (current + 1) << " (" << current->is_free << ", "
       << current->size << ") ";
    sep = " -> ";
    current = current->next;
  }
  std::cout << os.str() << "\n";
}

} // namespace kk

int main() {
  int *p1 = reinterpret_cast<int *>(kk::malloc(sizeof(int)));
  int *p2 = reinterpret_cast<int *>(kk::malloc(sizeof(int)));
  std::cout << "p1: " << p1 << "\n";
  std::cout << "p2: " << p2 << "\n";
  std::cout << "address of kk::malloc: " << (void *)(&kk::malloc) << "\n";
  std::cout << "address of p1: " << p1 << "\n";
  std::cout << "address of p2: " << p2 << "\n";
  kk::PrintBlockInfo();

  kk::free(p1);
  kk::PrintBlockInfo();

  p1 = reinterpret_cast<int *>(kk::malloc(sizeof(int)));
  kk::PrintBlockInfo();

  kk::free(p2);
  kk::PrintBlockInfo();

  double *d1 = reinterpret_cast<double *>(kk::malloc(sizeof(double)));
  kk::PrintBlockInfo();

  void *d2 = kk::malloc(0x10);
  kk::PrintBlockInfo();

  void *i = kk::malloc(4);
  kk::PrintBlockInfo();

  return 0;
}
/*


p1: 0x1b36018
p2: 0x1b36034
address of kk::malloc: 0x401057
address of p1: 0x1b36018
address of p2: 0x1b36034

0x1b36018 (0, 4)  -> 0x1b36034 (0, 4)

0x1b36018 (1, 4)  -> 0x1b36034 (0, 4)

0x1b36018 (0, 4)  -> 0x1b36034 (0, 4)

0x1b36018 (0, 4)  -> 0x1b36034 (1, 4)

0x1b36018 (0, 4)  -> 0x1b36034 (1, 4)  -> 0x1b36050 (0, 8)

0x1b36018 (0, 4)  -> 0x1b36034 (1, 4)  -> 0x1b36050 (0, 8)  -> 0x1b36070 (0, 16)

0x1b36018 (0, 4)  -> 0x1b36034 (0, 4)  -> 0x1b36050 (0, 8)  -> 0x1b36070 (0, 16)


*/
