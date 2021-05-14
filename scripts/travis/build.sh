#!/usr/bin/env bash

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
k2_dir=$(cd $cur_dir/../.. && pwd)

export cuda=$K2_CUDA_VERSION
export torch=$K2_TORCH_VERSION

which pyenv
pushd /opt/pyenv
git branch
git checkout master
popd

pyenv install ${K2_PYTHON_VERSION}.0
pyenv global ${K2_PYTHON_VERSION}.0
python3 --version
which python3


source $k2_dir/scripts/github_actions/install_cuda.sh

echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

nvcc --version

python3 -m pip install -q -U pip
python3 -m pip install -q wheel twine
python3 -m pip install -q bs4 requests tqdm

$k2_dir/scripts/github_actions/install_torch.sh
python3 -c "import torch; print('torch version:', torch.__version__)"
python3 -m torch.utils.collect_env

$k2_dir/scripts/github_actions/install_cudnn.sh

mkdir $k2_dir/build
pushd $k2_dir/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j2 _k2
