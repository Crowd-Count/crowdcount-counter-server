#!/bin/bash
# Device Registration Script
# Run this WHILE the server is running to register your devices

SERVER="http://localhost:5000"

echo "Registering devices..."

# Register T0001 as crowdcount1@crowdcount1 (developer counter - requires SSH password)
curl -X POST "$SERVER/device/register" \
  -H "Content-Type: application/json" \
  -d '{
    "stream_name": "T0001",
    "device_id": "crowdcount1@crowdcount1"
  }'
echo ""

# All other devices use default format (hostname=device_id, username=counter)
# Example for T0002:
# curl -X POST "$SERVER/device/register" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "stream_name": "T0002",
#     "device_id": "T0002"
#   }'

echo "✓ Devices registered!"
