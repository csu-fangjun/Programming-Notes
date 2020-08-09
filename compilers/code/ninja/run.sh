#!/bin/bash

rm -rf build
mkdir build
cd build
cmake -GNinja ..

# use 2 jobs
ninja -j 2
./main

cd ..
rm -rf build
