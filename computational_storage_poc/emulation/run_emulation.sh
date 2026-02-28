#!/bin/bash

echo "=========================================================="
echo " 🚀 Starting CHELATEDAI Computational Storage Emulator..."
echo "=========================================================="
# Start the privileged Docker container in the background
docker-compose up -d --build

echo "Waiting for Python FUSE to initialize the virtual Block device..."
sleep 4

echo -e "\n=========================================================="
echo " 🔍 TESTING OS SCSI READ INTERCEPTION IN SOFTWARE"
echo "=========================================================="
echo "Running usb_host_inference.py natively against the Docker virtual path (/mnt/virtual_usb/flash.img)..."
docker exec chelatedai_usb_emulator python3 /app/computational_storage_poc/usb_host_inference.py /mnt/virtual_usb/flash.img

echo -e "\n=========================================================="
echo " 📜 VIRTUAL SILICON EXECUTION LOGS (FUSE BACKEND)"
echo "=========================================================="
docker logs chelatedai_usb_emulator

echo -e "\n=========================================================="
echo " ✅ Emulation test complete. Shutting down container..."
echo "=========================================================="
docker-compose down
