#include <dlfcn.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void *(*pmalloc)(size_t size) = nullptr;
static void (*pfree)(void *ptr) = nullptr;

extern "C" void *malloc(size_t size) {
  if (pmalloc == nullptr) {
    pmalloc = reinterpret_cast<decltype(pmalloc)>(dlsym(RTLD_NEXT, "malloc"));
  }

  if (pmalloc == nullptr) {
    exit(0);
  }
  char s[] = "malloc called\n";
  write(2, s, sizeof(s));

  void *p = pmalloc(size);
  // return p;
  // std::cout << "p: " << p << std::endl;
  char a[30] = {0};
  snprintf(a, 30, "p is %p\n\n", p);
  write(2, a, sizeof(a));

  return p;
}

extern "C" void free(void *ptr) {
  char s[] = "free called\n";
  write(2, s, sizeof(s));
  if (pfree == nullptr) {
    pfree = reinterpret_cast<decltype(pfree)>(dlsym(RTLD_NEXT, "free"));
  }

  if (pfree == nullptr) {
    exit(0);
  }

  char a[30] = {0};
  snprintf(a, 30, "ptr is %p\n\n", ptr);
  write(2, a, sizeof(a));

  pfree(ptr);
}
