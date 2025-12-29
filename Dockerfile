FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /alpamayo

# Copy source code
COPY . .

# Install Python dependencies using uv
RUN uv sync --locked --no-dev --no-cache

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default command
CMD ["/bin/bash"]