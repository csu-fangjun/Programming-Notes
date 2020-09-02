#include <stdio.h>

// see `man tmpfile`

int main() {
  FILE *fp = tmpfile(); // it is opened with flags "w+b"

  char buf[] = "1234";
  int d = ftell(fp);
  printf("d: %d\n", d); // 0

  fwrite(buf, sizeof(char), sizeof(buf), fp);
  d = ftell(fp);
  printf("d: %d\n", d); // 5
  // note that the \0 character in "buf" is also written into fp

  rewind(fp);
  d = ftell(fp);
  printf("d: %d\n", d); // 0

  char s[10] = {0};
  int n = fread(s, sizeof(char), sizeof(s), fp);
  printf("n: %d\n", n); // n: 5
  printf("s: %s\n", s); // 1234

  fseek(fp, 0, SEEK_END);
  d = ftell(fp);
  printf("d: %d\n", d); // 5

  fclose(fp); // the tmp file is deleted
  return 0;
}
