# Start from the specified PyTorch base image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CPATH=/usr/local/cuda/include:$CPATH2

# Update and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libopenblas-dev \
    libopenexr-dev \
    libeigen3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-setuptools \
    python3-pip \
    libgl1-mesa-glx \
    libopenmpi-dev \
    mpich \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools==59.5.0
RUN conda install -c conda-forge mpi4py mpich -y
# Install Python packages
RUN pip install \
    pandas \
    tensorboard \
    numpy==1.24.1 \
    laspy \
    pytest \
    addict \
    pytorch-metric-learning==0.9.97 \
    yapf==0.40.1 \
    bitarray==1.6.0 \
    h5py \
    transforms3d \
    open3d \
    tqdm \
    einops\
    timm \
    termcolor
    

RUN pip install bagpy \
    utm \
    pyexr \
    pyntcloud \
    Jinja2==3.1.5 \
    ninja==1.11.1.1 \
    scikit-learn \
    torchtyping \
    future_fstrings \
    bitarray \
    pytorch_metric_learning==1.1.2 \
    psutil \
    tensorboardX \
    torchpack==0.3.1

RUN pip install torch_scatter==2.1.1 --verbose
RUN pip install torch_geometric==1.7.2 
RUN pip install spconv-cu116==2.3.6
RUN pip install torch_sparse==0.6.17
RUN pip install cumm-cu116==0.4.11
RUN pip install  torch_cluster==1.6.1
    
# Copy MinkowskiEngine source code into the image
COPY thirdparty/MinkowskiEngine /opt/MinkowskiEngine

# Build and install MinkowskiEngine
WORKDIR /opt/MinkowskiEngine
RUN python setup.py install --blas=openblas --force_cuda

COPY thirdparty/SparseTransformer /opt/SparseTransformer
WORKDIR /opt/SparseTransformer
RUN python setup.py install

# Clean up
WORKDIR /
RUN rm -rf /opt/MinkowskiEngine
RUN rm -rf /opt/SparseTransformer

# Set the default command
CMD ["bash"]
