#!/usr/bin/env bash

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
k2_dir=$(cd $cur_dir/.. && pwd)

echo "=============================="
echo " Display python version       "
echo "------------------------------"

K2_PYTHON_VERSION_LONG=3.8.0
export K2_PYTHON_VERSION=$(basename -s .0 $K2_PYTHON_VERSION_LONG)

export cuda=10.1
export torch=1.7.1

which pyenv
pushd /opt/pyenv
git branch
git checkout master
popd

pyenv install $K2_PYTHON_VERSION_LONG

echo "=============================="
echo " Display gcc version       "
echo "------------------------------"
gcc --version

# $k2_dir/

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

python3 -m pip install bs4 requests tqdm

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
