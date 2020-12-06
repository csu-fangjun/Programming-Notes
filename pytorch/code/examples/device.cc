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
 */

static void TestDeviceType() {
  // DeviceType is defined in namespace c10, but
  // it is exported to the namespace torch.
  torch::DeviceType device_type = torch::kCPU;
  std::string upper =
      torch::DeviceTypeName(device_type); // convert kCPU to a string
  std::string lower = torch::DeviceTypeName(device_type, true);
  std::cout << "device type: " << upper << "\n";
  std::cout << "device type: " << lower << "\n";
  std::cout << "device type: " << torch::kCPU << "\n"; // print kCPU directly
  /*
  device type: CPU
  device type: cpu
  device type: cpu
   */
}

static void TestDeviceImpl() {
  {
    std::cout << sizeof(torch::DeviceType) << "\n";  // 2
    std::cout << sizeof(c10::DeviceType) << "\n";    // 2
    std::cout << sizeof(torch::DeviceIndex) << "\n"; // 2
    std::cout << sizeof(torch::Device) << "\n";      // 4
  }
  {
    torch::Device device("cpu");
    std::cout << device << "\n"; // cpu
    std::string s = device.str();
    std::cout << s << "\n"; // cpu

    assert(device.is_cpu() == true);
    assert(device.type() == torch::kCPU);
    assert(device.index() == -1);
  }
  {
    torch::Device device("cuda:1");
    std::cout << device << "\n"; // cuda:1
    std::string s = device.str();
    std::cout << s << "\n"; // cuda:1

    assert(device.is_cuda() == true);
    assert(device.type() == torch::kCUDA);
    assert(device.index() == 1);
  }

  {
    torch::Device device(torch::kCUDA, 3);
    std::cout << device << "\n"; // cuda:3
    std::string s = device.str();
    std::cout << s << "\n"; // cuda:3

    assert(device.is_cuda() == true);
    assert(device.type() == torch::kCUDA);
    assert(device.index() == 3);
  }
}

void TestDevice() {
  //
  TestDeviceImpl();
  // TestDeviceType();
}
