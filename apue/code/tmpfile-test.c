#include <assert.h>
#include <stdio.h>
#include <string.h>

// see `man tmpfile`

int main() {
  // the returned file is unlinked, so if we close
  // fp, it is deleted.
  FILE *fp = tmpfile(); // it is opened with flags "w+b"

  char buf[] = "1234";
  int d = ftell(fp);
  assert(d == 0);

  fwrite(buf, sizeof(char), sizeof(buf), fp);
  d = ftell(fp);
  assert(d == 5);
  // note that the \0 character in "buf" is also written into fp

  rewind(fp);
  d = ftell(fp);
  assert(d == 0);

  char s[10] = {0};
  int n = fread(s, sizeof(char), sizeof(s), fp);
  assert(n == 5);
  assert(strcmp(s, "1234") == 0);

  fseek(fp, 0, SEEK_END);
  d = ftell(fp);
  assert(d == 5);

  fseek(fp, 0, SEEK_SET);
  d = ftell(fp);
  assert(d == 0);

  char a;
  n = fread(&a, sizeof(char), 1, fp);
  assert(n == 1);
  assert(a == '1');

  d = ftell(fp);
  assert(d == 1);
  a = '0';
  n = fwrite(&a, sizeof(char), 1, fp);
  assert(n == 1);

  // CAUTION: read and write share the same offset.
  // So fwrite advances the offset by 1
  d = ftell(fp);
  assert(d == 2);

  // now the file contains 1034

  n = fread(&a, sizeof(char), 1, fp);
  assert(n == 1);
  assert(a == '3');
  d = ftell(fp);
  assert(d == 3);

  fseek(fp, 1, SEEK_SET);

  d = ftell(fp);
  assert(d == 1);

  n = fread(&a, sizeof(char), 1, fp);
  assert(n == 1);
  assert(a == '0');
  d = ftell(fp);
  assert(d == 2);

  fclose(fp); // the tmp file is deleted
  return 0;
}
