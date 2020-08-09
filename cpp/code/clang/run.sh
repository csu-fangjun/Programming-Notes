#!/bin/bash

if [ ! -e compile_commands.json ]; then
  mkdir -p build
  cd build
  cmake ..

  ln -sfv $PWD/compile_commands.json ..
  cd ..
fi
clang-tidy test_clang_tidy.cc
