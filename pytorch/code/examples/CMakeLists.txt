cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(k2 CUDA CXX)

set(Python_ADDITIONAL_VERSIONS 3.8 3.6 3.7)
find_package(PythonInterp)
get_filename_component(PYTHON_DIR ${PYTHON_EXECUTABLE} DIRECTORY)

find_program(PYTHON_CONFIG_EXECUTABLE
  NAMES python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}-config
  PATHS ${PYTHON_INCLUDE_DIRS}
)
execute_process(
  COMMAND ${PYTHON_CONFIG_EXECUTABLE} "--includes"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
)
string(REPLACE " " ";" PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS})
list(GET PYTHON_INCLUDE_DIRS 0 PYTHON_INCLUDE_DIRS)
string(REPLACE "-I" "" PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS})

# or we can use `print(torch.utils.cmake_prefix_path)`
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_DIR
)

list(APPEND CMAKE_PREFIX_PATH "${TORCH_DIR}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_library(hello SHARED
  device.cc
  # cuda.cc
  # hello.cc
  intrusive_ptr_test.cc
  # tensor.cc
  )
set_property(TARGET hello PROPERTY CXX_STANDARD 14)

target_include_directories(hello PUBLIC
  ${TORCH_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)
target_compile_options(hello PUBLIC
  ${TORCH_CXX_FLAGS}
  -DTORCH_EXTENSION_NAME=hello
  -DTORCH_API_INCLUDE_EXTENSION_H
)
target_link_libraries(hello PUBLIC ${TORCH_LIBRARIES})

add_executable(main main.cc)
target_link_libraries(main hello)
