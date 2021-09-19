
#include "torch/torch.h"
#include <type_traits>

// see c10/core/ScalarType.h

// ScalarType is just an enum
//
// Note: os << ScalarType
// can print its name.

static void test_int_float_types() {
  // assert(torch::isIntegralType(torch::kBool) == false); // it prints warnings
  assert(torch::isIntegralType(torch::kBool, /*includeBool*/ true) == true);
  assert(torch::isIntegralType(torch::kByte, true) == true);
  assert(torch::isIntegralType(torch::kChar, true) == true);
  assert(torch::isIntegralType(torch::kShort, true) == true);
  assert(torch::isIntegralType(torch::kInt, true) == true);
  assert(torch::isIntegralType(torch::kLong, true) == true);
}

static void test_cpp_types() {
  assert(torch::CppTypeToScalarType<bool>::value == torch::kBool);
  assert(torch::CppTypeToScalarType<uint8_t>::value == torch::kByte);
  assert(torch::CppTypeToScalarType<int8_t>::value == torch::kChar);
  assert(torch::CppTypeToScalarType<int16_t>::value == torch::kShort);
  assert(torch::CppTypeToScalarType<int>::value == torch::kInt);
  assert(torch::CppTypeToScalarType<int64_t>::value == torch::kLong);
  assert(torch::CppTypeToScalarType<float>::value == torch::kFloat);
  assert(torch::CppTypeToScalarType<double>::value == torch::kDouble);

  assert(
      (std::is_same<typename c10::impl::ScalarTypeToCPPType<torch::kBool>::type,
                    bool>::value));
}

static void test_to_string() {
  // Note: torch::toString(torch::kBool) returns a `const char*`
  assert(torch::toString(torch::kBool) == std::string("Bool"));
  assert(torch::toString(torch::kByte) == std::string("Byte"));
  assert(torch::toString(torch::kChar) == std::string("Char"));
  assert(torch::toString(torch::kShort) == std::string("Short"));
  assert(torch::toString(torch::kInt) == std::string("Int"));
  assert(torch::toString(torch::kLong) == std::string("Long"));
  assert(torch::toString(torch::kFloat) == std::string("Float"));
  assert(torch::toString(torch::kDouble) == std::string("Double"));

  std::ostringstream os;
  os << torch::kBool;
  assert(os.str() == "Bool");
}

static void test_element_size() {
  assert(torch::elementSize(torch::kBool) == sizeof(bool));
  assert(torch::elementSize(torch::kByte) == sizeof(uint8_t));
  assert(torch::elementSize(torch::kChar) == sizeof(int8_t));
  assert(torch::elementSize(torch::kShort) == sizeof(int16_t));
  assert(torch::elementSize(torch::kInt) == sizeof(int));
  assert(torch::elementSize(torch::kLong) == sizeof(int64_t));
  assert(torch::elementSize(torch::kFloat) == sizeof(float));
  assert(torch::elementSize(torch::kDouble) == sizeof(double));
}

static void test() {
  test_element_size();
  test_to_string();
  test_cpp_types();
  test_int_float_types();
}

void test_scalar_type() { test(); }
