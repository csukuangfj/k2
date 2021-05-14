#!/usr/bin/env bash

df -h
free -m
echo "===================="
free -g

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
k2_dir=$(cd $cur_dir/../.. && pwd)

echo "k2_dir: $k2_dir"

export cuda=$K2_CUDA_VERSION
export torch=$K2_TORCH_VERSION

echo "========================================"
echo " Install Python ${K2_PYTHON_VERSION}.0  "
echo "----------------------------------------"

pushd /opt/pyenv
git checkout master
popd

pyenv install ${K2_PYTHON_VERSION}.0
pyenv global ${K2_PYTHON_VERSION}.0
python3 --version
which python3

echo "========================================"
echo " Install CUDA ${K2_CUDA_VERSION}        "
echo "----------------------------------------"
source $k2_dir/scripts/github_actions/install_cuda.sh

nvcc --version

echo "========================================"
echo " Install PyTorch ${K2_TORCH_VERSION}    "
echo "----------------------------------------"

python3 -m pip install -q -U pip
python3 -m pip install -q wheel twine
python3 -m pip install -q bs4 requests tqdm

$k2_dir/scripts/github_actions/install_torch.sh
python3 -c "import torch; print('torch version:', torch.__version__)"
python3 -m torch.utils.collect_env

echo "========================================"
echo " Install cuDNN 8.0                      "
echo "----------------------------------------"
$k2_dir/scripts/github_actions/install_cudnn.sh

echo "========================================"
echo " build k2                               "
echo "----------------------------------------"

echo "num_cpus: $(nproc)"

cd $k2_dir

export K2_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
export K2_MAKE_ARGS="-j2"
python setup.py bdist_wheel

echo "========================================"
echo " running test                           "
echo "----------------------------------------"

build_dir=$(find ./build -type d -name "*temp*")
echo "build_dir is: $(build_dir)"
cd $build_dir
make -j2
ctest --output-on-failure -j2
