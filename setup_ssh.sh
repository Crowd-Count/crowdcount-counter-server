#!/bin/bash
#
# SSH Setup Script for CrowdCount Device Management
# This script helps configure SSH access to your Raspberry Pi devices
#

set -e

echo "============================================================"
echo "CrowdCount Device Manager - SSH Setup"
echo "============================================================"
echo ""

# Check if SSH key exists
SSH_KEY="$HOME/.ssh/id_rsa"
if [ ! -f "$SSH_KEY" ]; then
    echo "No SSH key found at $SSH_KEY"
    echo "Generating new SSH key..."
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY" -N ""
    echo "✓ SSH key generated"
else
    echo "✓ SSH key already exists at $SSH_KEY"
fi

echo ""
echo "Now we'll copy your SSH key to your Raspberry Pi devices."
echo "You'll need to enter the Pi's password for each device."
echo ""

# Function to setup SSH for a device
setup_device() {
    local device_ip="$1"
    local device_name="$2"

    echo "Setting up SSH for $device_name ($device_ip)..."

    # Copy SSH key
    if ssh-copy-id -i "${SSH_KEY}.pub" "pi@${device_ip}"; then
        echo "✓ SSH key copied to $device_name"

        # Test connection
        if ssh -o BatchMode=yes -o ConnectTimeout=5 "pi@${device_ip}" "echo 'SSH connection successful'" &>/dev/null; then
            echo "✓ SSH connection verified for $device_name"
            return 0
        else
            echo "⚠ Warning: Could not verify SSH connection to $device_name"
            return 1
        fi
    else
        echo "✗ Failed to copy SSH key to $device_name"
        return 1
    fi
}

# Get list of Tailscale devices
echo "Fetching Tailscale devices..."
echo ""

if command -v tailscale &> /dev/null; then
    echo "Available Tailscale devices:"
    tailscale status | grep -v "^#" | head -10
    echo ""
else
    echo "⚠ Tailscale command not found. Make sure Tailscale is installed."
    echo ""
fi

# Prompt for device IPs
echo "Enter the Tailscale IP addresses of your Raspberry Pi devices."
echo "Press Enter after each IP. When done, press Enter on an empty line."
echo ""

devices=()
while true; do
    read -p "Enter Pi IP (or press Enter to finish): " device_ip

    if [ -z "$device_ip" ]; then
        break
    fi

    read -p "Enter device name (e.g., crowdcount1): " device_name

    if [ -z "$device_name" ]; then
        device_name="pi-${device_ip}"
    fi

    devices+=("$device_ip:$device_name")
    echo ""
done

if [ ${#devices[@]} -eq 0 ]; then
    echo "No devices entered. Exiting."
    exit 0
fi

echo ""
echo "Setting up SSH access for ${#devices[@]} device(s)..."
echo ""

success_count=0
for device in "${devices[@]}"; do
    IFS=':' read -r ip name <<< "$device"
    if setup_device "$ip" "$name"; then
        ((success_count++))
    fi
    echo ""
done

echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo "Successfully configured: $success_count/${#devices[@]} devices"
echo ""
echo "Next steps:"
echo "1. Start the counter server: python3 scripts/run_server.py"
echo "2. Open the web dashboard: http://localhost:5000"
echo ""
