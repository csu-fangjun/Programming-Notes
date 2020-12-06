#include <algorithm>
#include <iostream>
#include <vector>

// https://en.cppreference.com/w/cpp/algorithm/find

int main() {
  std::vector<int32_t> v = {1, 3, 2, 5, 6};
  auto it = std::find_if(v.begin(), v.end(), [](int32_t i) { return i == 5; });
  if (it != v.end())
    std::cout << "found!\n";
  else
    std::cout << "not found!\n";

  it = std::find(v.begin(), v.end(), 7);
  if (it != v.end())
    std::cout << "found! \n";
  else
    std::cout << "not found!\n";

  // the returned iterator points to the element that is not equal to 1
  it = std::find_if_not(v.begin(), v.end(), [](int32_t i) { return i == 1; });
  if (it != v.end())
    std::cout << "the element not equal to 1 is: " << *it << "\n";
  else
    std::cout << "it is : " << *it << "\n";

  return 0;
}
