cmake_minimum_required(VERSION 3.5)
project(hello)

# FetchContent is available from 3.11+
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})
include(FetchContent)

FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        master
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(pybind11)
endif()

list(APPEND CMAKE_MODULE_PATH "${pybind11_SOURCE_DIR}/tools")

include(pybind11Tools)

include_directories(${pybind11_SOURCE_DIR}/include)
pybind11_add_module(hello hello.cc)

add_custom_target(run-hello DEPENDS hello)
add_custom_command(
  TARGET run-hello
  POST_BUILD COMMAND python -c "import hello; print(hello.add(1, 2))"
  VERBATIM
  COMMENT "running hello"
)
