
Empty Base Optimization
=======================

Refer to:

  - `<https://en.cppreference.com/w/cpp/language/ebo>`_.
  - `<https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Empty_Base_Optimization>`_

.. literalinclude:: ./code/basics/ebo.cc
  :caption: ebo.cc
  :language: cpp
  :linenos:


.. Caution::

  Note that it is used in the implementation of ``std::unique_ptr``, which
  borrows ``compressed_pair`` from ``boost``.
