FROM python:3.10-slim

# Install system dependencies for FUSE and block loopbacks
RUN apt-get update && apt-get install -y \
    fuse \
    kmod \
    udev \
    && rm -rf /var/lib/apt/lists/*

# Install python fuse bindings
RUN pip install fusepy

# Make the virtual USB mount point
RUN mkdir -p /mnt/virtual_usb

WORKDIR /app
