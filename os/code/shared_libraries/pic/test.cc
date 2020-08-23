
static int s = 100;
int g = 10;
int test1() { return g + s; }
int test3();
int test2() { return g + test3(); }
int test3() { return g + test1(); }
