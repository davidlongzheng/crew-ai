#!/bin/bash

set -euxo pipefail

source venv/bin/activate
mkdir -p build

cd build
cmake ..
cmake --build .