Exception
=========

The page `<https://en.cppreference.com/w/cpp/error/exception>`_
lists all possible exception classes in STL.

``std::exception`` has a default constructor and copy constructor.
It has only one virtual function::

  virtual const char* what() const noexcept;

When it is default constructed, ``what()`` returns a implementation
defined string.

.. Note::

  ``what()`` is NOT pure virtual!


