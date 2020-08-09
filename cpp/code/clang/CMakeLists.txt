
cmake_minimum_required(VERSION 3.5)
project(clang_test)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(test_clang_tidy test_clang_tidy.cc)
