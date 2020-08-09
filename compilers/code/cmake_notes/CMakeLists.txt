cmake_minimum_required(VERSION 3.10)
project(Hello)

set(foo "hello-world")
string(REGEX REPLACE "-" "_" bar ${foo}) # from - to _
message(STATUS ${bar}) # now bar is hello_world
