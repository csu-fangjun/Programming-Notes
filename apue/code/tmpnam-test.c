#include <assert.h>
#include <stdio.h>

// never use tmpnam
//
// See the warning in `man tmpnam`

int main() {
  printf("L_tmpnam: %d, P_tmpdir: %s\n", L_tmpnam, P_tmpdir);
  // L_tmpnam: 20, P_tmpdir: /tmp

  // note that all the generated names have the prefix P_tmpdir
  char buf[L_tmpnam + 1];
  char *ret = tmpnam(buf);
  assert(ret == buf);
  printf("name is: %s\n", buf); // /tmp/fileLe0Q7N

  ret = tmpnam(buf);
  assert(ret == buf);
  printf("name is: %s\n", buf); // /tmp/filesD3HdL

  // ret points to a static buffer if we pass NULL
  ret = tmpnam(NULL);
  printf("ret: %p, %s\n", ret, ret); // 0x7f557b3d5700, /tmp/fileykcZ94

  ret = tmpnam(NULL);
  printf("ret: %p, %s\n", ret, ret); // 0x7f557b3d5700, /tmp/fileYpRakr

  return 0;
}
