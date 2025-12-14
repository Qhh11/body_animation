FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git pkg-config \
    ca-certificates wget \
    python3 python3-pip \
    libeigen3-dev \
    libceres-dev \
    libgoogle-glog-dev libgflags-dev \
    libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy
WORKDIR /workspace