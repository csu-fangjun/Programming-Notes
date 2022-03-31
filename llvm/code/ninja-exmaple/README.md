# usage

```bash
mkdir build
cd build
cmake -G Ninja ..
ninja -t targets
ninja
ninja clean
ninja -v
ninja clean
ninja -v hello
ninja -v main
./main

# Use cmake to build
# see https://cmake.org/cmake/help/git-stage/guide/user-interaction/index.html
ninja

# the options after -- are passed to ninja
cmake --build . -- -t targets

cmake --build . --target hello --verbose
```
