extern int test1();
extern int test2();
extern int test3();
extern int g;

int main() { return test1() + g + test2() + test3(); }
