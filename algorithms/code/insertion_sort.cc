#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void print(int* d, int n) {
  std::ostringstream os;
  std::string s = "";
  for (int i = 0; i < n; ++i) {
    os << s << d[i];
    s = ' ';
  }
  os << "\n";
  std::cout << os.str();
}

static void print(const std::vector<int>& v) {
  std::ostringstream os;
  std::string s = "";
  for (auto i : v) {
    os << s << i;
    s = ' ';
  }
  os << "\n";
  std::cout << os.str();
}

static void insertion_sort_v0(int* d, int n) {
  int i = 1;
  while (i < n) {
    int j = i;
    while (j > 0 && d[j - 1] > d[j]) {
      std::swap(d[j - 1], d[j]);
      --j;
    }
    ++i;
    print(d, n);
  }
}

static void insertion_sort_v1(int* d, int n) {
  int i = 1;
  while (i < n) {
    int j = i;
    int x = d[j];
    while (j > 0 && d[j - 1] > x) {
      d[j] = d[j - 1];
      --j;
    }
    d[j] = x;
    ++i;
    print(d, n);
  }
}

int main() {
  std::vector<int> v = {3, 7, 4, 9, 5, 2, 6, 1};
  print(v);
  insertion_sort_v1(v.data(), v.size());
  return 0;
}
