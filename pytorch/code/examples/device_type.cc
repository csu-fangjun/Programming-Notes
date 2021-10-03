
#include "torch/torch.h"

// see c10/core/DeviceType.h
//
// DeviceType is an enum
// We ususally only need to use:
//   torch::kCPU, torch::kCUDA
//
//  we can use "os << torch::kCPU" to print a device.

static void test() {
  // DeviceType is defined in namespace c10, but
  // it is exported to the namespace torch.
  {
    torch::DeviceType device_type = torch::kCPU;
    std::string upper =
        torch::DeviceTypeName(device_type); // convert kCPU to a string
    std::string lower = torch::DeviceTypeName(device_type, /*lower_case*/ true);

    assert(upper == "CPU");
    assert(lower == "cpu");
    std::ostringstream os;
    os << torch::kCPU;
    assert(os.str() == "cpu");
  }

  {
    torch::DeviceType device_type = torch::kCUDA;
    std::string upper =
        torch::DeviceTypeName(device_type); // convert kCUDA to a string
    std::string lower = torch::DeviceTypeName(device_type, /*lower_case*/ true);

    assert(upper == "CUDA");
    assert(lower == "cuda");
    std::ostringstream os;
    os << torch::kCUDA;
    assert(os.str() == "cuda");
  }
}

void test_device_type() { test(); }
