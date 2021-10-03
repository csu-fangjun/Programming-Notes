#include "torch/torch.h"

// see c10/core/StorageImpl.h
//
// StorageImpl is a subclass of c10::intrusive_ptr_target,
// so we can use c10::intrusive_ptr<c10::StorageImpl>.
//
// It contains:
//  a DataPtr, note it also contains a device
//  an allocator, maybe nullptr. Must not be null if resizable_ is True
//  size_bytes, number of bytes in the storage
//

static void test() {
  torch::Allocator *allocator = torch::GetAllocator(torch::kCPU);

  // torch::intrusive_ptr<torch::StorageImpl> storage_impl =
  auto storage_impl = torch::make_intrusive<torch::StorageImpl>(
      torch::StorageImpl::use_byte_size_t{},
      /*size_bytes*/ 20, allocator,
      /*resizable*/ true);
  // Since we passed an allocator and a size, it will allocate space internally
  // allocator.allocate() returns a DataPtr, which includes a torch::Device
  int32_t *data = (int32_t *)storage_impl->data();
  assert(storage_impl->nbytes() == 20);
  assert(storage_impl->resizable() == true);

  torch::DataPtr &p = storage_impl->data_ptr();
  assert(p.get() == data);
  assert(storage_impl->device() == torch::kCPU);
  assert(storage_impl->allocator() == allocator);
}

void test_storage_impl() { test(); }
