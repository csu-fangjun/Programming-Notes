#include <iostream>
#include <sstream>
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

static int partition(int* d, int low, int high) {
  int i = low;
  // [low, i-1]: less than d[pivot]
  // [i, high]: greater than or equal to d[pivot]
  int v = d[high];
  for (int k = low; k != high; ++k) {
    if (d[k] < v) {
      std::swap(d[i], d[k]);
      ++i;
    }
  }
  std::swap(d[i], d[high]);
  return i;
}

static void quick_sort(int* d, int low, int high) {
  if (low < high) {
    int pivot = partition(d, low, high);
    quick_sort(d, low, pivot - 1);
    quick_sort(d, pivot + 1, high);
  }
}

int main() {
  std::vector<int> v = {3, 7, 4, 9, 5, 2, 6, 1};
  print(v);
  quick_sort(v.data(), 0, v.size() - 1);
  print(v);
  return 0;
}
