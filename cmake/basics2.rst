
.. toctree::
  :maxdepth: 3

CMake
=====

CMakeCache.txt
--------------

set
---

Refer to the help doc `<https://cmake.org/cmake/help/v3.0/command/set.html>`_.

.. code-block::

  set(t2 "t2 value" CACHE STRING "help for t2")


.. code-block::

  cmake ..

produces the following in the ``CMakeCache.txt``::

  //help for t2
  t2:STRING=t2 value


If we change ``CMakeLists.txt``::

  set(t2 "t2 value2" CACHE STRING "help for t2")

and after running ``cmake ..``, the value for t2 is ``not``
changed since it exists already in the cache.

We have the following method to change ``t2``:

1. Change ``CMakeLists.txt`` to:

.. code-block::

  set(t2 "t2 value2" CACHE STRING "help for t2" FORCE)

2. Or use ``cmake -Dt2="t2 value2" ..``

Note that ``FORCE`` means to update the cache.
The command line option ``-Dt2`` also update the cache.
``cmake -Dt3=hello ..`` gives us a warning, saying that
`Manually-specified variables were not used by the project`;
but it will save ``t3`` to ``CMakeCache.txt``.

For ``set(t1 "hello")``, ``t1`` is not saved in the cache
file ``CMakeCache.txt``.

option
------

.. code-block::

  option(t2 "help info" "ON")
  option(t3 "default is OFF")

is equivalent to::

  set(t2 ON CACHE BOOL "help info")
  set(t3 OFF CACHE BOOL "default is OFF")

Note that the disadvantage of ``option`` is that
we cannot use ``FORCE`` in it!

list
----

.. code-block::

  set(src a.c b.c d.c)
  # set(src a.c;b.c;d.c)
  message(STATUS "${src}")

prints::

  a.c;b.c;d.c

.. code-block::

  set(src a.c;b.c;d.c)
  list(APPEND src hello.c)
  message(STATUS "${src}")

prints::

  a.c;b.c;d.c;hello.c


if
--

.. code-block::

    if (NOT DEFINED FOO)
      message(STATUS "FOO is not defined")
    else()
      message(STATUS "FOO is defined")
    endif()

    set(BAR ON)
    if (BAR)
      message(STATUS "bar is true")
    else()
      message(STATUS "bar is false")
    endif()

    set(BAR OFF)
    if (NOT BAR)
      message(STATUS "bar is false")
    else()
      message(STATUS "bar is true")
    endif()

    set(FOO "hello;world")
    if ("hello" IN_LIST FOO)
      message(STATUS "hello is found")
    else()
      message(STATUS "hello is not found")
    endif()

foreach
-------

.. code-block::

    set(FOO "hello;world")
    foreach(f ${FOO})
      message(STATUS "${f}")
    endforeach()

string
------

.. code-block::

    set(foo "hello")
    string(TOUPPER ${foo} bar)
    message(STATUS ${bar}) # now bar is HELLO

    set(foo "hello-world")
    string(REGEX REPLACE "-" "_" bar ${foo}) # from - to _
    message(STATUS ${bar}) # now bar is hello_world

CMAKE_BUILD_TYPE
----------------

.. code-block::

  if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
  endif()

CMAKE_CXX_FLAGS
---------------

.. code-block::

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


SHARED
------

.. code-block::

  add_library(example SHAED example.cc)
  set_target_properties(example PROPERTIES PREFIX "")
  set_target_properties(example PROPERTIES COMPILE_FLAGS "-std=c++11")
  set_target_properties(example PROPERTIES LINK_FLAGS "-lm")
  add_custom_command(TARGET example POST_BUILD COMMAND strip ${PROJECT_BINARY}/example.so)

  set_property(TARGET example PROPERTY BUILD_RPATH ${CMAKE_IMPLICIT_LINK_DIRECTORIES})
  set_target_properties(example PROPERTIES POSITION_INDEPENDENT_CODE ON)

