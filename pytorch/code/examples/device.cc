#include "torch/torch.h"
#include <iostream>
#include <string>

/*
Refer to
 - c10/core/Device.h
 - c10/core/DeviceType.h

A device has an index, which is of type int8_t. So the maximum
number of devices is 127.

A Device contains `DeviceIndex` and `DeviceType`.

If the device type is cpu, the device index has to be -1 or 0.

If the device type is CUDA, then a negative index means to use
the current device.

It method `str()` can return a string representation of the device,
e.g., "cpu", "cuda:0".

We can construct a device from its string representation.

The following methods are useful:

  - type()
  - index()
  - is_cpu()
  - is_cuda()
  - str()

Note: we can use "os << device;"
 */

static void test() {
  {
#if 0
    static_assert(sizeof(torch::DeviceType) == 1, "");
    static_assert(sizeof(torch::DeviceIndex) == 1, "");
    static_assert(sizeof(torch::Device) == 2, "");
#endif
    // for torch < 1.8.0
    // static_assert(sizeof(torch::DeviceType) == 2, "");
    // static_assert(sizeof(torch::DeviceIndex) == 2, "");
    // static_assert(sizeof(torch::Device) == 4, "");
  }
  {
    torch::Device device("cpu");
    assert(device.str() == "cpu");

    assert(device.is_cpu() == true);
    assert(device.type() == torch::kCPU);
    assert(device.index() == -1);
  }
  {
    torch::Device device("cuda:1");
    assert(device.str() == "cuda:1");

    assert(device.is_cuda() == true);
    assert(device.type() == torch::kCUDA);
    assert(device.index() == 1);
  }

  {
    torch::Device device(torch::kCUDA, 3);
    assert(device.str() == "cuda:3");

    assert(device.is_cuda() == true);
    assert(device.type() == torch::kCUDA);
    assert(device.index() == 3);
  }
}

void test_device() { test(); }
