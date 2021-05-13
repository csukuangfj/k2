#!/usr/bin/env bash

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
k2_dir=$(cd $cur_dir/.. && pwd)

echo "=============================="
echo " Display python version       "
echo "------------------------------"

which pyenv
pyenv install 3.8.0

python3 --version
which python3


echo "=============================="
echo " Display gcc version       "
echo "------------------------------"
gcc --version

# $k2_dir/

