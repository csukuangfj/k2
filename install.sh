#!/usr/bin/env bash
#
cuda=10.2
if false; then

# see https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/end-of-life/10.2/centos7/base/Dockerfile
# and
# https://github.com/OpenNMT/CTranslate2/blob/master/.github/workflows/ci.yml
cat > /etc/yum.repos.d/cuda.repo <<EOF
[cuda]
name=cuda
baseurl=https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA
EOF

export NVARCH=x86_64

export NVIDIA_GPGKEY_SUM=d0664fbbdb8c32356d45de36c5984617217b2d0bef41b93ccecd326ba3b80c87
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel7/${NVARCH}/D42D0685.pub |
  sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA

echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict  -
export CUDA_VERSION=10.2
export NV_CUDA_CUDART_VERSION=10.2.89-1
yum upgrade -y && yum install -y \
    cuda-cudart-10-2-${NV_CUDA_CUDART_VERSION} \
    cuda-compat-10-2 \
    && ln -s cuda-10.2 /usr/local/cuda \
    && yum clean all \
    && rm -rf /var/cache/yum/*

find / -name nvcc 2>/dev/null

export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
fi

which gcc
gcc --version

yum install centos-release-scl
yum install devtoolset-7-gcc-c++
which gcc
gcc --version

yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum --enablerepo=epel -y install cuda-10-2
export PATH=/usr/local/cuda-10.2/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH}

which nvcc
nvcc --version

export NV_CUDNN_VERSION=8.6.0.163-1
export NV_CUDNN_PACKAGE=libcudnn8-${NV_CUDNN_VERSION}.cuda10.2
export NV_CUDNN_PACKAGE_DEV=libcudnn8-devel-${NV_CUDNN_VERSION}.cuda10.2

yum install -y \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

python -m pip install torch==1.10.0


