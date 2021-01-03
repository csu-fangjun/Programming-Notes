// double free with delete
// gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex2.cc -o ex2
int main() {
  int *p = new int;
  delete p;
  delete p;
  return 0;
}
