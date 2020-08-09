#include <iostream>
#include <sstream>

int main() {
  int count;
  int ret = cudaGetDeviceCount(&count);
  std::cout << "device count: " << count << "\n"; // 8

  cudaDeviceProp prop;
  ret = cudaGetDeviceProperties(&prop, 0); // device 0
  std::ostringstream os;
  os << "device name: " << prop.name << "\n";                       // TITAN Xp
  os << "major.minor: " << prop.major << "." << prop.minor << "\n"; // 6.1
  os << "totalGlobalMem: " << prop.totalGlobalMem / 1024. / 1024. << " MB"
     << "\n"; // 12196.1 MB == 11.9 GB
  os << "sharedMemPerBlock: " << prop.sharedMemPerBlock / 1024. << " KB"
     << "\n"; // 48 KB
  os << "regsPerBlock: " << prop.regsPerBlock / 1024. << " KB"
     << "\n";                                  // 64 KB
  os << "warpSize: " << prop.warpSize << "\n"; // 32
  os << "memPitch: " << prop.memPitch / 1024. / 1024. / 1024. << " GB"
     << "\n";                                                      // 2 GB
  os << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << "\n"; // 1024
  os << "maxThreadsDim: " << prop.maxThreadsDim[0] << " "
     << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2]
     << "\n"; // 1024 1024 64
  os << "maxGridSize: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1]
     << " " << prop.maxGridSize[2] << "\n"; // 2147483647 65535 65535
  os << "multiProcessorCount: " << prop.multiProcessorCount << "\n"; // 30
  os << "maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor
     << "\n";                                                    // 2048
  os << "integrated: " << prop.integrated << "\n";               // 0
  os << "canMapHostMemory: " << prop.canMapHostMemory << "\n";   // 1
  os << "computeMode: " << prop.computeMode << "\n";             // 0
  os << "unifiedAddressing: " << prop.unifiedAddressing << "\n"; // 1
  os << "l2CacheSize: " << prop.l2CacheSize / 1024. / 1024. << "  MB"
     << "\n";                                                // 3 MB
  os << "managedMemory: " << prop.managedMemory << "\n";     // 1
  os << "isMultiGpuBoard: " << prop.isMultiGpuBoard << "\n"; // 0
  os << "clockRate: " << prop.clockRate / 1000. / 1000. << " MHz"
     << "\n"; // 1.582 MHz

  os << "\n";
  std::cout << os.str();

  return 0;
}
