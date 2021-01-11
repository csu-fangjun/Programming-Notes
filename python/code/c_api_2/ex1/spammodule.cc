// Refer to cpython/Modules/xxmodule.c, it is a template
// to be filled in for writing a new module from scratch.
//
// It is a convention that module `hello` is in file `hellomodule.c`.
#include <Python.h> // This is the first header.
                    // It should be included before other standard headers
                    // since it defines some macros inside it that affects
                    // the behavior of standard headers

// Python.h includes <stdio.h>, <string.h> <errno.h> and <stdlib.h>

static PyObject *spam_error;

static PyObject *SpamSystem(PyObject *self, PyObject *args) {
  // if `spam_system` is a module-level function, `self` points to the module
  // if `spam_system` is a method of an object, `self` points to the object
  //
  // `args` is a tuple
  const char *command; // use const since the returned string cannot be changed
  int sts;

  // PyArg_ParseTuple return true if every thing is parsed OK
  if (!PyArg_ParseTuple(args, "s", &command))
    return nullptr;

  sts = system(command);
  if (sts < 0) {
    PyErr_SetString(spam_error, "System command failed");
    return nullptr;
  }
  return PyLong_FromLong(sts);
}

// it can also be METH_VARAGS | METH_KEYWORDS, in this case the function
// receives an additional argument `PyObject *kwargs`
// and we should use `PyArg_ParseTupleAndKeywords`.
static PyMethodDef spam_methods[] = {
    {"system", SpamSystem, METH_VARARGS, "Execute a shell command"},
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef spammodule = {
    PyModuleDef_HEAD_INIT,
    "spam",        // module name
    "doc of spam", // module doc
    -1,
    spam_methods,
};

// PyMODINIT_FUNC is equivalent to `PyObject*`
// or `extern "C" PyObject*` if it is for C++.
//
// `import spam` in Python will first invoke `PyInit_spam`
//
// The returned value is inserted into `sys.modules`
PyMODINIT_FUNC PyInit_spam() {
  PyObject *m;
  m = PyModule_Create(&spammodule);
  if (m == nullptr)
    return nullptr;

  // the Python name for the exception object is `spam.error`
  spam_error = PyErr_NewException("spam.error", nullptr, nullptr);
  // we should check that spam_error is not NULL !
  Py_INCREF(spam_error);

  PyModule_AddObject(m, "error", spam_error);
  return m;
}
