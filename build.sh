#!/bin/bash

set -euxo pipefail

source venv/bin/activate
mkdir -p build

conan install . --output-folder=build --build=missing 

cd build
VENV=../venv
export CMAKE_PREFIX_PATH=$VENV/lib/python3.12/site-packages/pybind11/share/cmake
export CC=/opt/homebrew/bin/gcc-14
export CXX=/opt/homebrew/bin/g++-14
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .