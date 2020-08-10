
#include "gperftools/tcmalloc.h"
#include <iostream>

int main() {
  int major;
  int minor;
  const char *patch;
  patch = tc_version(&major, &minor, nullptr);
  std::cout << "major: " << major << "\n";
  std::cout << "minor: " << minor << "\n";
  std::cout << "patch: " << patch << "\n";

  return 0;
}
