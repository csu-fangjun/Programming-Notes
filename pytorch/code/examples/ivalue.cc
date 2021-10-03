// ivalue is first introduced in
//  https://github.com/pytorch/pytorch/pull/9368
//  The merged commit is 9ed2190bdb2784e500b39d4e2d855625a2db2ad6
//
// the next commit is a949245a8694eef39789d8346362e2cec7d59966
//  see https://github.com/pytorch/pytorch/pull/9718/files
//
// the next commit is f62bc01dfee69acbd936a452c48bcdeccc9dd408
// https://github.com/pytorch/pytorch/pull/9575
//
// the next commit is e39c8043dc97950318ff83eb8502c52f8f4b63d9
// https://github.com/pytorch/pytorch/pull/9763
//
// the next commit is 5e5c15dd426ae608868ea76ec5be2d8817a31523
// https://github.com/pytorch/pytorch/pull/9948
// It adds tensorlist
//
// the next commit is 170d29769b6f31f191a99461f7e7e51f005b8a8d
// It adds `ConstantString``.
//
// the next commit is f7b02b3a685a7721307c776f17dffda763966641
// https://github.com/pytorch/pytorch/pull/10824
// It uses c10::intrusive_ptr in ivalue
//
//  0a6931cfee93a4c70d17980786337799ed5d56ee moves ivalue.h from torch/csrc/jit
//  to aten
//
//  e8ecbcdf010d1e65384ba2d1f8760cc557c02883
//  https://github.com/pytorch/pytorch/pull/11610
//  It moves ivalue from jit to aten
//
// d1ac1eba3b53f67d8d12eb20002b06893a2a4d2e
// https://github.com/pytorch/pytorch/pull/11834
// It adds bool to ivalue
//
// 4e1c64caee5037917f603c3683973667238c0163
// https://github.com/pytorch/pytorch/pull/12582
// It supports torch::optional
//
// 4d62eef505770f1c866fa2205a5a05ab47865dda
// https://github.com/pytorch/pytorch/pull/12976
// It adds Future to ivalue
//
// 0fd176fea45abb092c903b40008c7119e21ee9f5
// https://github.com/pytorch/pytorch/pull/13336
// It adds is, is not, to ivalue
//
// 78d594f46cf48cff620aa3e69c40697882a503f3
// https://github.com/pytorch/pytorch/pull/14666
// It adds torch::Device to ivalue
//
// 3f8fd19a868eb4aa1154744fb484e7e2cb555aec
// https://github.com/pytorch/pytorch/pull/16208
// It adds dict to ivalue
#include "torch/torch.h"

namespace {
struct Foo : public torch::CustomClassHolder {
  int a;
  Foo(int v) : a(v) {}
  int get_a() const { return a; }
};

} // namespace

// see torch/custom_class.h
static auto register_foo = torch::class_<Foo>("myclasses", "Foo");

static void test_custom_class() {
#if 0
  torch::intrusive_ptr<torch::CustomClassHolder> foo =
      torch::make_intrusive<Foo>(10);
  torch::IValue v(std::move(foo));
#else
  torch::IValue v = torch::make_custom_class<Foo>(10);
#endif
  assert(v.isCustomClass() == true);

  torch::intrusive_ptr<Foo> p = v.toCustomClass<Foo>();
  assert(p->get_a() == 10);
}

static void test_int() {
  torch::IValue v(10);
  std::cout << "v: " << v << "\n";
  // v: 10
  assert(v.isInt() == true);
  assert(v.toInt() == 10);
}

static void test() {
  test_custom_class();
  test_int();
  //
}

void test_ivalue() { test(); }
