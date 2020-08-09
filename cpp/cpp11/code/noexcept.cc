#include <exception>
#include <iostream>
#include <string>

namespace {

class MyException : public std::exception {
 public:
  explicit MyException(const std::string& msg) : msg_(msg) {}
  const char* what() const noexcept override { return msg_.c_str(); }

 private:
  std::string msg_;
};

void t1() { throw MyException{"throw in t1"}; }
void t2() noexcept(false) { throw MyException{"throw in t2"}; }

void t3() noexcept {
  // the caller is unable to catch the exception thrown in this function
  // since it is marked as `noexcept`.
  // `noexcept` means that the function will not throw,
  // if it indeed throws, the user cannot catch it and `std::terminate()`
  // is called immediately
  throw MyException{"std::terminate() is called"};
}

void test1() {
  try {
    t1();
  } catch (const std::exception& ex) {
    std::cout << "caught: " << ex.what() << "\n";
  }
  // print:
  //   caught: throw in t1
}

void test2() {
  try {
    t2();
  } catch (const std::exception& ex) {
    std::cout << "caught: " << ex.what() << "\n";
  }
  // print:
  //  caught: throw in t2
}

void test3() {
  try {
    t3();
  } catch (const std::exception& ex) {
    // it will never catch the exception in `t3` since
    // it is marked as `noexcept`
    std::cout << "caught: " << ex.what() << "\n";
  }
  // terminate called after throwing an instance of '(anonymous
  // namespace)::MyException'
  //  what(): std::terminate() is called
  // Aborted (core dumped)
}

void test() {
  test1();
  test2();
  test3();
}

}  // namespace

int main() {
  test();
  return 0;
}
