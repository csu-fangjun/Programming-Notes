#include <cassert>
#include <cstdint>
#include <iostream>
#include <pthread.h>

pthread_key_t key;
int32_t value;

void Destructor(void *p) {
  std::cout << "deleting:" << p << "\n";
  delete (int32_t *)p;
}

void Init() {
  int32_t ret = pthread_key_create(&key, nullptr);
  assert(ret == 0);
  std::cout << "value is: " << &value << "\n";
  pthread_setspecific(key, nullptr);
}

void Display() {
  int32_t *v = (int32_t *)pthread_getspecific(key);
  std::cout << "v: " << v << "\n";
}

void *Run(void *) {
  int32_t *v = (int32_t *)pthread_getspecific(key);
  std::cout << "v is: " << v << "\n";
  if (v == nullptr) {
    v = new int32_t;
    std::cout << "new v is: " << v << "\n";
    pthread_setspecific(key, v);
  }
  Display();
  return nullptr;
}
int main() {
  Init();
  pthread_t p[2];
  for (int32_t i = 0; i != 2; ++i) {
    pthread_create(&p[i], nullptr, &Run, nullptr);
  }

  for (int32_t i = 0; i != 2; ++i) {
    pthread_join(p[i], nullptr);
  }
  pthread_key_delete(key);
  return 0;
}
