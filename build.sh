#!/bin/bash

set -euxo pipefail

source venv/bin/activate
mkdir -p build
cd build

VENV=../venv
export CMAKE_PREFIX_PATH=$VENV/lib/python3.12/site-packages/pybind11/share/cmake
cmake ../
make