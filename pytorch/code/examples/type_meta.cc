#include "torch/torch.h"

// see c10/util/typeid.h
//
// TypeMeta contains only an index_, which indexes into
// a static member array variable.
//
// It supports:
//   - converting a ScalarType to TypeMeta
//   - converting a TypeMeta to ScalarType

static void test() {
  // requires 1.8.0
  static_assert(sizeof(caffe2::TypeMeta) == sizeof(uint16_t));
  caffe2::TypeMeta t = caffe2::TypeMeta::Make<int32_t>();
  caffe2::TypeMeta t2 = caffe2::TypeMeta::fromScalarType(torch::kInt);
  assert(t == t2);

  // convert a TypeMeta to a scalar type
  assert(t.toScalarType() == torch::kInt);

  assert(t == torch::kInt);
  assert(t != torch::kFloat);

  assert(t.Match<int32_t>());
  assert(t.Match<int8_t>() == false);

  assert(t.name() == "int");
  assert(t.itemsize() == 4);

  assert(t.isScalarType(torch::kInt) == true);
  assert(t.isScalarType(torch::kShort) == false);

  assert(t.isScalarType() == true);

  assert(torch::scalarTypeToTypeMeta(torch::kInt) == t);
  assert(torch::typeMetaToScalarType(t) == torch::kInt);
}

void test_type_meta() { test(); }
