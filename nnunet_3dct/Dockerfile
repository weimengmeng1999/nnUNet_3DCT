FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

ARG USER_ID
ARG GROUP_ID
ARG USER

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# install CUDA-enabled PyTorch first (matches base image CUDA version)
# This is the official PyTorch install for CUDA 11.8
RUN pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Optional: set environment vars for nnUNet
# ENV nnUNet_raw="/nfs/home/mwei/nnUNet_3d_data/nnUNet_raw" \
#     nnUNet_preprocessed="/nfs/home/mwei/nnUNet_3d_data/nnUNet_preprocessed" \
#     nnUNet_results="/nfs/home/mwei/nnUNet_3d_data/nnUNet_results_wcls"
    
RUN addgroup --gid $GROUP_ID $USER && \
    adduser --disabled-password --gecos "" --uid $USER_ID --gid $GROUP_ID $USER

WORKDIR /nfs/home/$USER/nnUNet_3DCT/nnunet_3dct

COPY . .

RUN pip install -e .

CMD ["/bin/bash"]

