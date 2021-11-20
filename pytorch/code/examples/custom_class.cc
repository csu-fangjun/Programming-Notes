#include "torch/script.h"

// It must be a subclass of CustomClassHolder.
// Otherwise, it results in compile time error.
struct Foo : torch::CustomClassHolder {
  int i = 0;
  Foo(int k) : i(k) {}
};

class Bar : public torch::CustomClassHolder {};

static void test() {
  torch::class_<Foo> f("k2", "Foo");
  // we can get a class ptr of a custom class by:
  //
  // its fully qualified name, e.g., "__torch__.torch.classes.k2.Foo"
  // or
  // by its type, e.g., torch::intrusive_ptr<Foo>

  assert(torch::isCustomClassRegistered<Foo>() == false);
  // we have to use torch::intrusive_ptr<Foo>
  assert(torch::isCustomClassRegistered<torch::intrusive_ptr<Foo>>());

  torch::ClassTypePtr p =
      torch::getCustomClassType<torch::intrusive_ptr<Foo>>();

  assert(p->name()->qualifiedName() == "__torch__.torch.classes.k2.Foo");

  torch::TypePtr p2 = torch::getTypePtr<torch::intrusive_ptr<Foo>>();
  assert(p->str() == p2->str());
  assert(p->kind() == p2->kind());
  assert(p == p2);

  assert(p == torch::getCustomClass(p->name()->qualifiedName()));

  torch::ClassTypePtr q =
      torch::ClassType::create(torch::QualifiedName("k2.Bar"),
                               std::weak_ptr<torch::jit::CompilationUnit>());
  assert(q->name()->qualifiedName() == "k2.Bar");

  assert(torch::isCustomClassRegistered<torch::intrusive_ptr<Bar>>() == false);
  torch::getCustomClassTypeMap().insert(
      {std::type_index(typeid(torch::intrusive_ptr<Bar>)), q});
  assert(torch::isCustomClassRegistered<torch::intrusive_ptr<Bar>>() == true);

  torch::registerCustomClass(q);

  // Note: there are two maps
  //
  // one for CustomClass, key: name, value: torch::ClassTypePtr
  // (in custom_class.cpp)
  //
  // one for CustomClassTypeMap, key: std::type_index, value:
  // torch::ClassTypePtr
  // (in ivalue.cpp)
  const_cast<torch::QualifiedName &>(p->name().value()) =
      torch::QualifiedName("k2.Foo");
  assert(p->name()->qualifiedName() == "k2.Foo");
  assert(p2->cast<torch::ClassType>()->name()->qualifiedName() == "k2.Foo");
  // We need to re-register the class type since we changed its name.
  torch::registerCustomClass(p);

  // Use of custom class
  //
  // Example 1
  torch::intrusive_ptr<Foo> fp = torch::make_intrusive<Foo>(100);
  torch::IValue fival(std::move(fp));
  // see ivalue_inl.h
  // The above constructor set the tag to Tag::Object
  assert(fival.isObject());
  // so what is saved in `fival`?
  // It create a torch::ivalue::Object (not in torch/csrc/jit/api/object.h)
  // (it is in ivalue_inl.h)
  // It saves a copy of the class ptr of Foo inside the Object.
  // Also, it uses IValue::make_capsule to create a capsule to save `fp`.

  // Note: the kind of fival is `Object`.
  {
    torch::intrusive_ptr<Foo> k = fival.toCustomClass<Foo>();
    assert(k->i == 100);
  }
}

void test_custom_class() { test(); }
