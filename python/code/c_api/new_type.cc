#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

// clang-format off
struct Person {
  PyObject_HEAD
  double ob_fval;
};
// clang-format on

static void PrintType(PyTypeObject* p) {
  std::ostringstream os;
#define PRINT(x) os << #x << ": " << p->x << "\n"
  PRINT(ob_base.ob_base.ob_refcnt);
  PRINT(ob_base.ob_base.ob_type);  // even if we did not set its type, it is set
                                   // to &PyType_Type
  PRINT(ob_base.ob_size);          // 0
  PRINT(tp_name);                  // 0
  PRINT(tp_basicsize);             // sizeof(Person)
  PRINT(tp_itemsize);              // 0
  PRINT(tp_dealloc);         // object_dealloc, PyBaseObject_Type tp_dealloc
  PRINT(tp_getattr);         // 0
  PRINT(tp_setattr);         // 0
  PRINT(tp_as_async);        // 0
  PRINT(tp_repr);            // object_repr, PyBaseObject_Type tp_repr
  PRINT(tp_as_number);       // 0
  PRINT(tp_as_sequence);     // 0
  PRINT(tp_as_mapping);      // 0
  PRINT(tp_hash);            // _Py_HashPointer, PyBaseObject_Type tp_hash
  PRINT(tp_call);            // 0
  PRINT(tp_str);             // object_str, PyBaseObject_Type tp_str
  PRINT(tp_getattro);        // PyObject_GenericGetAttr, from base
  PRINT(tp_setattro);        // PyObject_GenericSetAttr, from base
  PRINT(tp_as_buffer);       // 0
  PRINT(tp_flags);           // 266240 = 0x41000
  PRINT(tp_doc);             // doc of hello.Person
  PRINT(tp_traverse);        // 0
  PRINT(tp_richcompare);     // object_richcompare, from base
  PRINT(tp_weaklistoffset);  // 0
  PRINT(tp_iter);            // 0
  PRINT(tp_iternext);        // 0
  PRINT(tp_methods);         // 0
  PRINT(tp_members);         // 0
  PRINT(tp_getset);          // 0
  PRINT(tp_base);            // &PyBaseObject_Type
  PRINT(tp_dict);            // not NULL!
  PRINT(tp_descr_get);       // 0
  PRINT(tp_descr_set);       // 0
  PRINT(tp_dictoffset);      // 0
  PRINT(tp_init);            // object_init, tp_init from PyBaseObject_Type
  PRINT(tp_alloc);       // PyType_GenericAlloc, tp_alloc from PyBaseObject_Type
  PRINT(tp_new);         // we set it manually to PyType_GenericNew
  PRINT(tp_free);        // PyObject_Del, tp_free from PyBaseObject_Type
  PRINT(tp_is_gc);       // 0
  PRINT(tp_bases);       // not NULL !
  PRINT(tp_mro);         // not NULL !
  PRINT(tp_cache);       // 0
  PRINT(tp_subclasses);  // 0
  PRINT(tp_weaklist);    // not NULL !
  PRINT(tp_del);         // 0
  PRINT(tp_version_tag);  // 0
  PRINT(tp_finalize);     // 0

  std::cout << os.str() << "\n";
  std::cout << (void*)p->tp_repr << "\n";
  std::cout << (void*)(PyBaseObject_Type.tp_repr) << "\n";
}

static void AddNewType(PyObject* m) {
  // it is never freed.
  PyTypeObject* PersonType = new PyTypeObject;
  std::memset(PersonType, 0, sizeof(PyTypeObject));
  Py_XINCREF(PersonType);
  assert(Py_REFCNT(PersonType) == 1);

  PersonType->tp_name = "hello.Person";
  PersonType->tp_doc = "doc of hello.Person";
  PersonType->tp_basicsize = sizeof(Person);
  PersonType->tp_itemsize = 0;
  PersonType->tp_flags = Py_TPFLAGS_DEFAULT;

  // this is important; otherwise we cannot create it!
  // if its base class is not NULL and is not PyBaseObject_Type,
  // then set it to NULL will inherit the tp_new from the parent class
  PersonType->tp_new = PyType_GenericNew;

  if (PyType_Ready(PersonType) < 0) {
    PyErr_SetString(PyExc_RuntimeError, "TypeReady: PersonType");
    return;
  }

  // not its type is PyType_Type
  assert(Py_TYPE(PersonType) == &PyType_Type);

  // its base is PyBaseObject_Type
  assert(PersonType->tp_base == &PyBaseObject_Type);
  assert(PersonType->tp_init == PyBaseObject_Type.tp_init);
  assert(PersonType->tp_alloc == PyBaseObject_Type.tp_alloc);
  assert(PersonType->tp_alloc == &PyType_GenericAlloc);
  assert(PersonType->tp_free == PyBaseObject_Type.tp_free);
  assert(PersonType->tp_free == &PyObject_Del);

  assert(PersonType->tp_dealloc == PyBaseObject_Type.tp_dealloc);
  assert(PersonType->tp_repr == PyBaseObject_Type.tp_repr);
  assert(PersonType->tp_hash == PyBaseObject_Type.tp_hash);
  assert(PersonType->tp_str == PyBaseObject_Type.tp_str);

  assert(PersonType->tp_dict != nullptr);
  // tp_dict is unique!
  assert(PersonType->tp_dict != PyBaseObject_Type.tp_dict);

  assert(PersonType->tp_getattro == PyBaseObject_Type.tp_getattro);
  assert(PersonType->tp_getattro == &PyObject_GenericGetAttr);

  assert(PersonType->tp_setattro == PyBaseObject_Type.tp_setattro);
  assert(PersonType->tp_setattro == &PyObject_GenericSetAttr);

  assert(PersonType->tp_richcompare == PyBaseObject_Type.tp_richcompare);

  PyModule_AddObject(m, "Person", (PyObject*)PersonType);

  // PrintType(PersonType);
}

void TestNewType(PyObject* m) { AddNewType(m); }
