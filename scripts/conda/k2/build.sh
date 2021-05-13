#!/usr/bin/env bash
#
# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

set -ex

CONDA_ENV_DIR=$CONDA_PREFIX
if [ $K2_IS_GITHUB_ACTIONS -eq 1 ]; then
  NUM_JOBS="-j2"
else
  NUM_JOBS="-j"
fi

echo "K2_PYTHON_VERSION: $K2_PYTHON_VERSION"
echo "K2_TORCH_VERSION: $K2_TORCH_VERSION"
echo "K2_CUDA_VERSION: $K2_CUDA_VERSION"
echo "K2_BUILD_TYPE: $K2_BUILD_TYPE"
echo "K2_BUILD_VERSION: $K2_BUILD_VERSION"
python3 --version

echo "CC is: $CC"
echo "GCC is: $GCC"
echo "which nvcc: $(which nvcc)"
echo "gcc version: $($CC --version)"
echo "nvcc version: $(nvcc --version)"

python3 setup.py install --single-version-externally-managed --record=record.txt
