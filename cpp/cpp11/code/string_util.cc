#include <cassert>
#include <iostream>
#include <string>

struct SourceLocation {
  const char* file;
  const char* function;
  int32_t line;
};

std::ostream& operator<<(std::ostream& os, const SourceLocation& loc) {
  os << loc.function << " at " << loc.file << ":" << loc.line;
  return os;
}

// input: /foo/bar/abc.txt
// output: abc.txt
std::string Basename(const std::string& path) {
  const char kSep = '/';
  size_t pos = path.rfind(kSep);
  if (pos != std::string::npos) {
    return path.substr(pos + 1);
  }
  return path;
}

void test() {
  auto s = Basename("/foo/bar/a.txt");
  assert(s == "a.txt");

  SourceLocation loc{"this.cc", "test", __LINE__};
  std::cout << loc << "\n";
}

int main() {
  test();
  return 0;
}
