cmake_minimum_required(VERSION 3.5)
add_library(hello SHARED hello.cc)
add_executable(main main.cc)
target_link_libraries(main PRIVATE hello)

