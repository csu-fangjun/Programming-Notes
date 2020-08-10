#include <dlfcn.h>
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

  return pmalloc(size);
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

  pfree(ptr);
}
