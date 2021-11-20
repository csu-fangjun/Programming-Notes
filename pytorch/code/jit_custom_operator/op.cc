// see https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html

#include "torch/script.h"
#include <vector>

#include "ops.h"

// we can use torch.classes.foo.MyStack later in Python
TORCH_LIBRARY(foo, m) {
  m.class_<MyStack>("MyStack")
      .def(torch::init<std::vector<int64_t>>())
      .def("top",
           [](const c10::intrusive_ptr<MyStack> &self) {
             return self->v.back();
           })
      .def("push", &MyStack::push)
      .def("pop", &MyStack::pop)
      .def("clone", &MyStack::clone)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<MyStack> &self) -> std::vector<int64_t> {
            return self->v;
          },
          // __setstate__
          [](std::vector<int64_t> state) -> c10::intrusive_ptr<MyStack> {
            return c10::make_intrusive<MyStack>(state);
          });
}
