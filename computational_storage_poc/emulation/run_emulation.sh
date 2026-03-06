#!/bin/bash
set -euo pipefail

if docker compose version >/dev/null 2>&1; then
  compose_cmd=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  compose_cmd=(docker-compose)
else
  echo "Docker Compose is required but was not found." >&2
  exit 1
fi

cleanup() {
  "${compose_cmd[@]}" down
}

echo "=========================================================="
echo " 🚀 Starting CHELATEDAI Computational Storage Emulator..."
echo "=========================================================="
"${compose_cmd[@]}" up -d --build
trap cleanup EXIT

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
echo " ✅ Emulation test complete."
echo "=========================================================="
