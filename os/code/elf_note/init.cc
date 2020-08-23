
class Foo {
public:
  Foo() { a_ = 12345; }
  ~Foo() { a_ = 54321; }
  int Get() const { return a_; }
  int a_;
};

Foo f;

int test() { return f.Get(); }
