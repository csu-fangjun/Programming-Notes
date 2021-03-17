
Alignment
=========

- ``alignof(T)``
- ``alignas(T)``
- ``std::alignment_of<T>``
- ``std::aligned_storage<sizeof(T), alignof(T)>``

.. code-block:: cpp

  template <class _Tp> struct alignment_of
      : public integral_constant<size_t, alignof(_Tp)> {};


.. literalinclude:: ./code/align.cc
  :caption: align.cc
  :language: cpp
  :linenos:
