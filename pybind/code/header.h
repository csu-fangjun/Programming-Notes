#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<int>);
