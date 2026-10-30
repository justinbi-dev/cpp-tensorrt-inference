# Base image: matches your JetPack/L4T version.
# Check your host's L4T version with: cat /etc/nv_tegra_release
# and pick a matching tag from nvcr.io/nvidia/l4t-jetpack
FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    libopencv-dev \
    libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# TensorRT, CUDA, and cuDNN headers/libs come from the l4t-jetpack base image,
# so no separate install step is needed for those.

WORKDIR /app

# --- Copy source ---
# Copy CMakeLists first for layer caching if you add deps later,
# then copy the rest of the source.
COPY CMakeLists.txt .
COPY src/ ./src/
COPY include/ ./include/
# add any config/model dirs your app expects at runtime, e.g.:
# COPY config/ ./config/
# COPY models/ ./models/

# --- Build ---
RUN mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j$(nproc)

# --- Runtime ---
# Adjust to your actual binary name/path from CMakeLists
ENTRYPOINT ["./build/orin_detector"]`