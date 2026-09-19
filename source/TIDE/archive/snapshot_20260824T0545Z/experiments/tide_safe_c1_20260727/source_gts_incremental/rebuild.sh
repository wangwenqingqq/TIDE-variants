#!/usr/bin/env bash
set -euo pipefail

rm -rf bin build

mkdir -p bin build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
cd ..