#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cassert>
#include <cstring>
#include <iostream>

/*
In Include/methodobject.h, it has

typedef PyObject* (*PyCFunction)(PyObject*, PyObject*);
typedef PyObject* (*PyCFunctionWithKeywords)(PyObject*, PyObject*, PyObject*);
typedef PyObject* (*PyCMethod)(PyObject*, PyObject*, PyObject* const*, size_t,
PyObject*);

struct PyMethodDef {
        const char* ml_name;
        PyCFunction ml_meth;
        int ml_flags;
        const char* ml_doc;
};

typedef struct PyMethodDef PyMethodDef;

In Include/cpython/methodobject.h, it has

typedef struct {
        PyObject_HEAD
        PyMethodDef* m_ml; // borrowed
        PyObject* m_self; // borrowed
        PyObject* m_module; // borrowed
        PyObject* m_weakreflist;
        vectorcallfunc vectorcall;
} PyCFunctionObject;

typedef struct {
        PyCFunctionObject func;
        PyTypeObject *mm_class;
} PyCMethodObject;

*/

static PyObject* Say(PyObject* self, PyObject* args) {
  // we will set it to a module method, so `self` should be NULL.
  assert(self == NULL);

  return PyLong_FromLong(10);

#if 0
  Py_RETURN_NONE;
#elif 0
  Py_INCREF(Py_None);
  return Py_None;
#endif
}

void AddMethod(PyObject* m) {
  PyModule_Check(m);

  // TODO(fangjun): how to free PyMethodDef ???
  PyMethodDef* f = new PyMethodDef;
  std::memset(f, 0, sizeof(PyMethodDef));
  f->ml_name = "say";
  f->ml_meth = reinterpret_cast<PyCFunction>(&Say);
  f->ml_flags = METH_VARARGS;

  // it returns a PyCFunctionObject, defined in Include/cpython/methodobject.h
  PyObject* o = PyCFunction_NewEx(f, NULL, NULL);
  assert(PyCFunction_Check(o) == 1);

  // Objects/methodobject.c

  assert(Py_TYPE(o) == &PyCFunction_Type);
  assert(o->ob_type == &PyCFunction_Type);
  assert(strcmp(o->ob_type->tp_name, "builtin_function_or_method") == 0);
  // its `tp_call` is `cfunction_call`
  /*
  cfunction_call is in Objects/methodobject.c

  static PyObject* cfunction_call(PyObject* func, PyObject*args,
  PyObject*kwargs);
  */

  assert(PyCFunction_GetFunction(o) == f->ml_meth);
  assert(PyCFunction_GetFlags(o) == f->ml_flags);
  // this is not a method and not static, so self is NULL
  assert(PyCFunction_GetSelf(o) == NULL);

  assert(PyCallable_Check(o) == 1);
  assert(Py_TYPE(o)->tp_call != NULL);

  assert(Py_REFCNT(o) == 1);
  PyModule_AddObject(m, "say", o);  // it steals the reference for o
  assert(Py_REFCNT(o) == 1);
}

static PyObject* SayWithSelf(PyObject* self, PyObject* args) {
  // since we pass a unicode string while creating this method,
  // so the following assert should be true
  assert(PyUnicode_Check(self) == 1);
  std::cout << "in method: " << self << "\n";
  Py_RETURN_NONE;
}

static void AddMethodWithSelf(PyObject* m) {
  auto self = PyUnicode_FromString("world");
  assert(Py_REFCNT(self) == 1);

  PyMethodDef* f = new PyMethodDef;
  f->ml_name = "say_self";
  f->ml_meth = reinterpret_cast<PyCFunction>(&SayWithSelf);
  f->ml_doc = "help for say self";
  f->ml_flags = METH_VARARGS;
  PyObject* o = PyCFunction_New(f, self);  // `self` ref cnt is increased by one
  assert(Py_REFCNT(self) == 2);

  PyObject* t = PyCFunction_GetSelf(o);  // it is NOT a new reference
  assert(Py_REFCNT(self) == 2);          // it is still 2

  // const char* s = PyUnicode_AsUTF8(t);
  // std::cout << "self is " << s << "\n";   // self is world
  std::cout << t << " " << self << "\n";  // 0x7f1c2a0935f0 0x7f1c2a0935f0
  // note that t and self points to the same object!
  //
  assert(Py_REFCNT(o) == 1);
  // if we register it as `say2`, then we can call it with `module.say2`
  // but since the  ml_name is say_self, `help(model.say2)` displays
  // the function name as `say_self`!
  // PyModule_AddObject(m, "say2", o);  // it steals the reference for o
  PyModule_AddObject(m, "say_self", o);  // it steals the reference for o
  assert(Py_REFCNT(o) == 1);
}

void TestMethods(PyObject* m) {
  AddMethod(m);
  AddMethodWithSelf(m);
}
