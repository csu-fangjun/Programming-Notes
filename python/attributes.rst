Attributes
==========

.. literalinclude:: ./code/c_api_2/sample_code/attributes.cc
  :caption: sample_code/attributes.cc
  :language: cpp
  :linenos:

PyObject_SetAttr
----------------

``int PyObject_SetAttr(PyObject *o, PyObject *attr_name, PyObject *v)``

- ``setattrfunc PyTypeObject.tp_setattr``

    ``int (*setattrfunc)(PyObject *self, char *attr, PyObject *value)``

- ``setattrofunc PyTypeObject.tp_setattro``

    ``int (*setattrofunc)(PyObject *self, PyObject *attr, PyObject *value)``





