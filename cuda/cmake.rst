
CMake
=====

Build cuda with CMake.

.. code-block::

  cmake -DCMAKE_CUDA_FLAGS="-arch=sm_30" ..

  set_target_properties(example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

.. code-block::

  # refer to
  # https://on-demand.gputechconf.com/gtc/2017/presentation/S7438-robert-maynard-build-systems-combining-cuda-and-machine-learning.pdf

  cmake_minimum_required(VERSION 3.8)
  project(Example CUDA CXX)
  if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
  endif()

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED 11)
    set(CMAKE_CXX_EXTENSIONS OFF)
  endif()

  add_library(example SHARED a.cc b.cu d.cu)

  target_include_directories(example
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hello
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hello>
      $<INSTALL_INTERFACE:include/hello>
  )

  target_compile_definitions(example
    PRIVATE KW_EXPORTS
    INTERFACE KW_IMPORTS
  )

  target_link_libraries(example
    PUBLIC foo_bar
  )


Interface library:

.. code-block::

  add_library(example INTERFACE)
  target_include_directories(example INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hello>
    $<INSTALL_INTERFACE:include/hello>
  )

  set(cxx_lang "$<COMPILE_LANGUAGE:CXX>")
  set(cuda_lang "$<COMPILE_LANGUAGE:CUDA>")
  set(debug_cxx_lang "$<AND:$<CONFIG:DEBUG>,${cxx_lang}>")
  set(debug_cxx_lang "$<AND:$<CONFIG:DEBUG>,${cuda_lang}>")

  target_compile_options(example INTERFACE
    # build flags we want for all CXX builds
    $<${cxx_lang}:$<BUILD_INTERFACE:-Wall>>

    # build flags we want for all CUDA builds
    $<${cuda_lang}:$<BUILD_INTERFACE:-Xcompiler=-Wall>>

    # build flags we want for all CXX debug builds
    $<${debug_cxx_lang}:$<BUILD_INTERFACE:-Wshadow;-Wunused-parameter>>

    # build flags we want for all CUDA debug builds
    $<${debug_cuda_lang}:$<BUILD_INTERFACE:-Xcompiler=-Wshadow,-Wunused-parameter>>
  )

Separable compilation: `<https://on-demand.gputechconf.com/gtc/2017/presentation/S7438-robert-maynard-build-systems-combining-cuda-and-machine-learning.pdf>`_

.. code-block::

  add_library(example STATIC abc.cu)
  set_target_properties(example
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    PROPERTIES POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(example PRIVATE hello)

References
----------

- `<https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html>`_
- Refer to `<https://gitlab.kitware.com/robertmaynard/cmake_cuda_tests/-/tree/master/>`
  for examples.
- Build Systems: Combining CUDA and Modern CMake `<https://on-demand.gputechconf.com/gtc/2017/presentation/S7438-robert-maynard-build-systems-combining-cuda-and-machine-learning.pdf>`_
