
__attribute__((section("foo"))) void test() {}
int a = 10;
int b;
void bar() {}
int main() {
  asm volatile("mov $0x01, %eax\n"
               "mov $0x10, %ebx\n"
               "int $0x80\n");
  // since we do not link the  c runtime library,
  // we use exit system call.
}

#if 0
readelf -h main # it shows that the entry point is indeed main

objdum -d main # it shows that the address of main is 0x10006
nm main # it shows that the address of main is 0x10006

readelf -l main # it shows that segment 00 is comprised of .text foo

readelf -S main # show sections

readelf -e main # equvialent to -h -S -l
#endif
