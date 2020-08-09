#include <setjmp.h>

#include <iostream>

void test1(jmp_buf env, int i, int* b) {
  std::cout << "test1 called with " << i << "\n";
  *b += 2;
  longjmp(env, i);
  std::cout
      << "this line is not reachable. It jumps to the place "
      << "where env is initialized, i.e., jump to where  setjmp was invoked. "
      << "setjmp returns the value i passed to longjmp"
      << "\n";
}

void test() {
  int b = 10;
  std::cout << "before calling test1, b is " << b << "\n";
  jmp_buf env;
  int r = setjmp(env);
  if (r == 3) {
    // it is returned from longjmp since we passed 3 to longjmp
    std::cout << "after returning from longjmp, b is " << b << "\n";
    // Note that b is changed inside test1.
    // what setjmp saves is just some pointers, e.g., IP, SP, BP.
    // It does NOT save the whole stack!
    //
    // If test is returned, then env is no long valid!
  } else if (r == 0) {
    test1(env, 3, &b);
  } else {
    std::cout << "unknown return code from setjmp!"
              << "\n";
  }
}

int main() {
  test();
  return 0;
}
