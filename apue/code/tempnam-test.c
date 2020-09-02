#include <stdio.h>
#include <stdlib.h>

// Never use tempnam
//
// See the warning in `man tempnam`

int main() {
  char *f =
      tempnam("/mydir", "123456"); // only 5 bytes in the prefix are effective

  printf("f: %s\n", f); // f: /tmp/12345Thb39B

  free(f); // we have to free it

  f = tempnam("/etc", "1234");
  printf("f: %s\n", f); // f: /etc/12341TFp2I

  free(f); // we have to free it

  f = tempnam("/bin", "1");
  printf("f: %s\n", f); // f: /bin/1wgT9qM

  free(f); // we have to free it

  f = tempnam("/bin", "");
  printf("f: %s\n", f); // f: /bin/filevSZb50

  free(f);

  f = tempnam("./abc", ""); // first we create the directory: mkdir ./abc
  printf("f: %s\n", f);     // f: ./abc/filewf8Mb8

  free(f);

  return 0;
}
