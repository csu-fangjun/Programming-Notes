#pragma once

#include "torch/script.h"
#include <vector>
struct MyStack : public torch::CustomClassHolder {
  explicit MyStack(const std::vector<int64_t> &val) : v(val) {}
  void push(int64_t x) { v.push_back(x); }
  int64_t pop() {
    auto i = v.back();
    v.pop_back();
    return i;
  }
  c10::intrusive_ptr<MyStack> clone() const {
    return c10::make_intrusive<MyStack>(v);
  }

  std::vector<int64_t> v;
};
