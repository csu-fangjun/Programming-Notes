#include <cassert>
#include <initializer_list>
#include <iostream>
#include <vector>

#include <torch/torch.h>

// refer to https://pytorch.org/cppdocs/notes/tensor_basics.html#cpu-accessors
static void test_accessor() {
  torch::Tensor t = torch::ones({3, 3}, torch::kFloat32);
  // assert its data type is float and ndim is 2
  auto accessor = t.accessor<float, 2>();

  float trace = 0;
  for (int i = 0; i < accessor.size(0); ++i) {
    trace += accessor[i][i];
  }

  assert(trace == 3);
}

// refer to
// https://pytorch.org/cppdocs/notes/tensor_basics.html#using-externally-created-data
static void test_external() {
  float d[] = {0, 1, 2, 3, 4, 5};
  torch::Tensor t = torch::from_blob(d, {2, 3});
}

// refer to
// https://pytorch.org/cppdocs/notes/tensor_creation.html#specifying-a-size
static void test_sizes() {
  {
    // 1-d tensor
    torch::Tensor t = torch::ones(2);
    assert(t.size(0) == 2);
    assert(t.size(-1) == 2);

    assert(t.sizes()[0] == 2);
    assert(t.sizes() == (std::vector<int64_t>{2}));
  }

  {
    // 2-d tensor
    torch::Tensor t = torch::zeros({2, 3});
    assert(t.size(0) == 2);
    assert(t.size(1) == 3);

    assert(t.size(-1) == 3);
    assert(t.size(-2) == 2);

    assert(t.sizes()[0] == 2);
    assert(t.sizes()[1] == 3);

    assert(t.sizes() == (std::vector<int64_t>{2, 3}));
  }
}

static void test_tensor_options() {
  torch::TensorOptions options =
      torch::TensorOptions()
          .dtype(torch::kFloat32)  // default: kFlaot32
          .layout(torch::kStrided) // default:kStrided
          .device(torch::kCUDA, 0) // default: kCPU
          .requires_grad(true);    // default: false

  auto t = torch::empty({2, 3}, options);
  assert(t.dtype() == torch::kFloat32);
  assert(t.layout() == torch::kStrided);
  assert(t.device().type() == torch::kCUDA);
  assert(t.device().index() == 0);
  assert(t.requires_grad() == true);

  t = torch::empty(2, torch::dtype(torch::kInt32)
                          .layout(torch::kStrided)
                          .device(torch::kCPU)
                          .requires_grad(false));
  assert(t.dtype() == torch::kInt32);
  assert(t.layout() == torch::kStrided);
  assert(t.device().type() == torch::kCPU);
  assert(t.requires_grad() == false);
}

static void test_ref() {
  auto t = torch::zeros(3);
  assert(t.use_count() == 1);

  {
    auto t2 = t;
    assert(t.use_count() == 2);
    assert(t2.use_count() == 2);
  }
  assert(t.use_count() == 1);
}

static void test_device_type() {
  static_assert(sizeof(torch::kCPU) == sizeof(int8_t), "");
  static_assert(sizeof(torch::kCUDA) == sizeof(int8_t), "");
}

static void test_device() {
  static_assert(sizeof(torch::Device) == (sizeof(int8_t) + sizeof(int8_t)), "");

  static_assert(sizeof(torch::Device) ==
                    (sizeof(torch::DeviceType) + sizeof(torch::DeviceIndex)),
                "");

  {
    torch::Device device(torch::kCPU);
    assert(device.type() == torch::kCPU);
    assert(device.has_index() == false);
    assert(device.index() == -1);
    assert(device.is_cpu() == true);
    assert(device.is_cuda() == false);
    assert(std::string("cpu") == device.str());

    std::ostringstream os;
    os << device;
    assert(os.str() == "cpu");
  }
  {
    torch::Device device("cpu");
    assert(device.type() == torch::kCPU);
    assert(device.has_index() == false);
    assert(device.index() == -1);
    assert(device.is_cpu() == true);
    assert(device.is_cuda() == false);
    assert(std::string("cpu") == device.str());
  }

  // now for cuda
  {
    torch::Device device(torch::kCUDA, 3);
    assert(device.type() == torch::kCUDA);
    assert(device.has_index() == true);
    assert(device.index() == 3);
    assert(device.is_cpu() == false);
    assert(device.is_cuda() == true);
    assert(std::string("cuda:3") == device.str());

    device.set_index(4);
    assert(device.index() == 4);
    assert(std::string("cuda:4") == device.str());
  }

  {
    torch::Device device("cuda:2");
    assert(device.type() == torch::kCUDA);
    assert(device.has_index() == true);
    assert(device.index() == 2);
    assert(device.is_cpu() == false);
    assert(device.is_cuda() == true);
    assert(std::string("cuda:2") == device.str());
  }
}

static void test_data_ptr() {
  {
    // default constructor
    torch::DataPtr ptr;
    assert(ptr == nullptr);
    assert(ptr.device() == torch::Device("cpu"));

    assert(ptr.get() == nullptr);
    assert(ptr.get_context() == nullptr);
  }
  {
    // case 1: DataPtr does not manage the passed ptr since
    // no context is passed
    float *p = new float;
    torch::DataPtr ptr(p, {"cpu"});

    assert(ptr != nullptr);
    assert(ptr.device() == torch::Device("cpu"));
    assert(ptr.get() == p);
    assert(ptr.get_context() == nullptr);

    delete p;
  }

  {
    float *p = new float;
    printf("p is: %p\n", p);
    torch::DataPtr ptr(p, /*context*/ p,
                       [](void *q) {
                         printf("deleting %p\n", q);
                         delete reinterpret_cast<float *>(q);
                       },
                       {"cpu"});
  }
}

static void test_scalar() {
  {
    torch::Scalar scalar;
    assert(scalar.isIntegral(/*includeBool*/ true) == true);
    assert(scalar.to<int>() == 0);
  }
  {
    torch::Scalar scalar(1);
    assert(scalar.isIntegral(/*includeBool*/ true) == true);
    assert(scalar.to<int>() == 1);
  }

  {
    torch::Scalar scalar(1.25);
    assert(scalar.isIntegral(/*includeBool*/ true) == false);
    assert(scalar.to<double>() == 1.25);
  }
}

static void test_int_array_ref() {
  // note that
  // using IntArrayRef = ArrayRef<int64_t>;
  //
  // ArrayRef is like string_view;
  //
  static_assert(std::is_same<torch::IntArrayRef::value_type, int64_t>::value,
                "");
  {
    // from a variable
    int64_t i = 1;
    torch::IntArrayRef a(i);
    assert(a.size() == 1);
    assert(a[0] == i);
    i = 2;
    assert(a[0] == i);
  }

  {
    // from a initializer_list
    torch::IntArrayRef a({1, 2, 3});
    // Note that {1, 2, 3} is saved in a static array variable!
    assert(a.size() == 3);

    // it has overloadded opartor=
    assert(a == (std::vector<int64_t>{1, 2, 3}));
  }
}

void test_hello() {
  // test_int_array_ref();

  // test_scalar();
  // test_data_ptr();
  // test_device_type();
  // test_accessor();
  // test_external();
  test_sizes();
  // test_tensor_options();
  // test_ref();

  // std::cout << t.device() << "\n";
  // std::cout << t << "\n";
  // t = t.to(torch::kCUDA);
  //
  // std::cout << t << "\n";
}
