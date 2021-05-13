#!/usr/bin/env bash

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
k2_dir=$(cd $cur_dir/../.. && pwd)

echo "=============================="
echo " Display python version       "
echo "------------------------------"

export cuda=$K2_CUDA_VERSION
export torch=1.7.1

which pyenv
pushd /opt/pyenv
git branch
git checkout master
popd

pyenv install ${K2_PYTHON_VERSION}.0
pyenv global ${K2_PYTHON_VERSION}.0
python3 --version
which python3

echo "=============================="
echo " Display gcc version       "
echo "------------------------------"
gcc --version


echo "=============================="
echo " Install CUDA toolkit $cuda   "
echo "------------------------------"

source $k2_dir/scripts/github_actions/install_cuda.sh

echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

nvcc --version

echo "=============================="
echo " Install torch $torch         "
echo "------------------------------"

python3 -m pip install -q -U pip
python3 -m pip install -q wheel twine
python3 -m pip install -q bs4 requests tqdm

$k2_dir/scripts/github_actions/install_torch.sh
python3 -c "import torch; print('torch version:', torch.__version__)"
python3 -m torch.utils.collect_env

echo "=============================="
echo " Download cuDNN 8.0           "
echo "------------------------------"

$k2_dir/scripts/github_actions/install_cudnn.sh

mkdir $k2_dir/build
pushd $k2_dir/build
cmake -DCMAKE_BUILD_TYPE=Release ..
cat k2/csrc/version.h
make VERBOSE=1 -j _k2
