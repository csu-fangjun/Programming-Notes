#include "torch/script.h"

// see aten/src/ATen/core/qualified_name.h

static void test1() {
  torch::QualifiedName q("foo");
  assert(q.qualifiedName() == "foo");

  assert(q.prefix() == "");
  assert(q.name() == "foo");
}

static void test2() {
  torch::QualifiedName q("foo.bar");
  assert(q.qualifiedName() == "foo.bar");

  assert(q.prefix() == "foo");
  assert(q.name() == "bar");
}

static void test() {
  test1();
  test2();
}

void test_qualified_name() { test(); }
