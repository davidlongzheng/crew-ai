#!/bin/bash

set -euxo pipefail

if [ -d "venv" ]; then
    source venv/bin/activate
fi
mkdir -p build

cd build
cmake ..
cmake --build .