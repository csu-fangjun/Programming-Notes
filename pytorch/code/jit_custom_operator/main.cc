#include "ops.h"
#include "torch/csrc/jit/frontend/parser.h"
#include "torch/csrc/jit/serialization/import_source.h"
#include "torch/script.h"
#include "unpickler.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

class VectorReader : public caffe2::serialize::ReadAdapterInterface {
public:
  VectorReader(const std::vector<char> &data) : data_(std::move(data)) {}

  size_t size() const override { return data_.size(); }

  size_t read(uint64_t pos, void *buf, size_t n,
              const char *what) const override {
    std::copy(data_.data() + pos, data_.data() + pos + n,
              reinterpret_cast<char *>(buf));
    return n;
  }

private:
  std::vector<char> data_;
};

using TypeResolver =
    std::function<c10::StrongTypePtr(const c10::QualifiedName &)>;

using ObjLoader = std::function<c10::intrusive_ptr<c10::ivalue::Object>(
    at::StrongTypePtr, torch::IValue)>;

torch::IValue readArchiveAndTensors(
    const std::string &archive_name, c10::optional<TypeResolver> type_resolver,
    c10::optional<ObjLoader> obj_loader, c10::optional<at::Device> device,
    caffe2::serialize::PyTorchStreamReader &stream_reader) {
  std::string picklename = archive_name + ".pkl";
  at::DataPtr pickle_ptr;
  size_t pickle_size;
  std::tie(pickle_ptr, pickle_size) = stream_reader.getRecord(picklename);

  size_t bytes_read = 0;
  auto data = reinterpret_cast<const char *>(pickle_ptr.get());
  auto reader = [&](char *buffer, size_t len) -> size_t {
    if (bytes_read >= pickle_size) {
      return 0;
    }
    len = std::min(pickle_size - bytes_read, len);
    // Copy len bytes into buffer
    const char *start = data + bytes_read;
    std::memcpy(buffer, start, len);
    bytes_read += len;
    return len;
  };

  std::string archive_name_plus_slash = archive_name + "/";
  auto read_record = [&](const std::string &name) {
    std::string ss = archive_name_plus_slash + name;
    return std::get<0>(stream_reader.getRecord(ss));
  };

  torch::jit::Unpickler2 unpickler(
      reader, type_resolver ? std::move(*type_resolver) : nullptr,
      obj_loader ? std::move(*obj_loader) : nullptr, std::move(read_record),
      device);
  unpickler.set_version(stream_reader.version());
  return unpickler.parse_ivalue();
}

struct Foo : public torch::CustomClassHolder {
  int i;
};

int32_t main() {

  std::string qualClassName = "_k2.Foo";
  at::ClassTypePtr classTypePtr =
      at::ClassType::create(c10::QualifiedName(qualClassName),
                            std::weak_ptr<torch::jit::CompilationUnit>());

  classTypePtr->addAttribute("capsule", at::CapsuleType::get());

  c10::getCustomClassTypeMap().insert(
      {std::type_index(typeid(c10::intrusive_ptr<Foo>)), classTypePtr});
  c10::getCustomClassTypeMap().insert(
      {std::type_index(typeid(c10::tagged_capsule<Foo>)), classTypePtr});

  torch::jit::registerCustomClass(classTypePtr);

  torch::jit::Parser parser(std::make_shared<torch::jit::Source>("_k2.Foo"));

  std::cout << "type " << torch::jit::getCustomClass("_k2.Foo") << "\n";
  // return 0;

  torch::jit::SourceImporter importer(
      std::make_shared<torch::jit::CompilationUnit>(), nullptr, nullptr, 2);
  std::cout << "type: " << importer.loadType("_k2.Foo")->str() << "\n";

  return 0;

  auto module = torch::jit::load("ab2.pt");
  c10::intrusive_ptr<MyStack> iv =
      module.attr("stack").toCustomClass<MyStack>();
  std::cout << iv->pop() << "\n";
  std::cout << iv->pop() << "\n";
  return 0;

  FILE *fp = fopen("ab.pt", "rb");
  assert(fp != nullptr);

  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  std::vector<char> f(size);
  size_t len = fread(f.data(), 1, size, fp);
  assert(len == size);

  fclose(fp);

  caffe2::serialize::PyTorchStreamReader reader(
      std::make_unique<VectorReader>(f));

  auto ivalue = readArchiveAndTensors("data",
                                      /*class_resolver=*/c10::nullopt,
                                      /*obj_loader=*/c10::nullopt,
                                      /*device=*/c10::nullopt, reader);
  std::cout << ivalue.tagKind() << "\n";

  return 0;
}
