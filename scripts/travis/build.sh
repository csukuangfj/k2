#!/usr/bin/env bash

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
k2_dir=$(cd $cur_dir/.. && pwd)

echo "=============================="
echo " Display python version       "
echo "------------------------------"

K2_PYTHON_VERSION_LONG=3.8.0
export K2_PYTHON_VERSION=$(basename -s .0 $K2_PYTHON_VERSION_LONG)

which pyenv

pyenv install --list
pushd /opt/pyenv
git branch
git checkout master
git pull
git log -3
popd
pyenv install --list
#
pyenv install $K2_PYTHON_VERSION_LONG
python3 --version
which python3


echo "=============================="
echo " Display gcc version       "
echo "------------------------------"
gcc --version

# $k2_dir/

