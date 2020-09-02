#include <stdio.h>
#include <stdlib.h>

int main() {
  char template[] = "1234567abcXXXXXX";

  // char template[] = "XXXXXx"; // error! the last 6 characters MUST be XXXXXX
  int fd = mkstemp(template);

  // it creates a file 1234567abciehxSY4
  // the file is NOT removed automatically after closing it.
  if (fd == -1) {
    perror("mkstemp");
    printf("failed !\n");
    return 0;
  }

  printf("fd: %d\n", fd);
  printf("template: %s\n", template);

  FILE *fp = fdopen(fd, "r+b");

  fwrite(template, sizeof(char), sizeof(template), fp);

  fclose(fp);
  return 0;
}
