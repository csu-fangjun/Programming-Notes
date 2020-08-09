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

static void bubble_sort_v0(int* d, int n) {
  bool swapped = true;
  while (swapped) {
    swapped = false;
    for (int i = 0; i < n - 1; ++i) {
      if (d[i] > d[i + 1]) {
        std::swap(d[i], d[i + 1]);
        swapped = true;
      }
    }
    if (swapped) {
      print(d, n);
    }
  }
}

static void bubble_sort_v1(int* d, int n) {
  bool swapped = true;
  for (int i = 1; swapped && i < n; ++i) {
    swapped = false;
    for (int j = 0; j < n - i; ++j) {
      if (d[j] > d[j + 1]) {
        std::swap(d[j], d[j + 1]);
        swapped = true;
      }
    }
    if (swapped) {
      print(d, n);
    }
  }
}

static void bubble_sort_v2(int* d, int n) {
  bool swapped = true;
  int m = n;
  while (swapped) {
    swapped = false;
    for (int i = 0; i < m - 1; ++i) {
      if (d[i] > d[i + 1]) {
        std::swap(d[i], d[i + 1]);
        swapped = true;
      }
    }
    --m;
    if (swapped) {
      print(d, n);
    }
  }
}

static void bubble_sort_v3(int* d, int n) {
  int new_n = n;
  while (new_n > 1) {
    int k = 0;
    for (int i = 0; i < new_n - 1; ++i) {
      if (d[i] > d[i + 1]) {
        std::swap(d[i], d[i + 1]);
        k = i + 1;
      }
    }
    new_n = k;
    print(d, n);
  }
}

int main() {
  std::vector<int> v = {9, 5, 2, 4, 1, 8};
  print(v);
  bubble_sort_v3(v.data(), v.size());
  return 0;
}
