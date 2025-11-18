#!/usr/bin/env python3
"""
Test script for device registration API
Demonstrates how to register device information with the server
"""

import requests
import json
import sys


def register_device(server_url, stream_name, device_id, password=""):
    """
    Register a device with the counter server

    Args:
        server_url: Base URL of the counter server (e.g., http://localhost:5000)
        stream_name: Name of the stream
        device_id: Unique device identifier
        password: Optional password for authentication
    """
    endpoint = f"{server_url}/device/register"

    payload = {
        "stream_name": stream_name,
        "device_id": device_id,
        "password": password
    }

    try:
        print(f"Registering device: {device_id}")
        print(f"Stream name: {stream_name}")
        print(f"Has password: {bool(password)}")
        print()

        response = requests.post(endpoint, json=payload)

        if response.status_code == 200:
            result = response.json()
            print("✓ Registration successful!")
            print(f"  Stream: {result.get('stream_name')}")
            print(f"  Device ID: {result.get('device_id')}")
        else:
            print(f"✗ Registration failed: {response.status_code}")
            print(f"  Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to server at {server_url}")
        print("  Make sure the server is running")
    except Exception as e:
        print(f"✗ Error: {e}")


def get_server_status(server_url):
    """Get server status including device info"""
    endpoint = f"{server_url}/status"

    try:
        response = requests.get(endpoint)

        if response.status_code == 200:
            status = response.json()
            print("\n=== Server Status ===")
            print(f"Running: {status.get('running')}")
            print(f"MediaMTX Healthy: {status.get('mediamtx_healthy')}")
            print(f"Active Streams: {status.get('active_streams')}")
            print(f"Processors: {status.get('processor_count')}/{status.get('max_processors')}")

            device_info = status.get('device_info', {})
            if device_info:
                print("\n=== Registered Devices ===")
                for stream, info in device_info.items():
                    print(f"  {stream}:")
                    print(f"    Device ID: {info.get('device_id')}")
                    print(f"    Has Password: {info.get('has_password')}")
            else:
                print("\nNo devices registered")

        else:
            print(f"✗ Failed to get status: {response.status_code}")

    except Exception as e:
        print(f"✗ Error getting status: {e}")


def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 test_device_registration.py <server_url> <stream_name> <device_id> [password]")
        print()
        print("Examples:")
        print("  python3 test_device_registration.py http://localhost:5000 crowdcount1 entrance-camera-01")
        print("  python3 test_device_registration.py http://localhost:5000 crowdcount1 entrance-camera-01 mypassword")
        print()
        print("After registration, check the server status to see registered devices.")
        sys.exit(1)

    server_url = sys.argv[1]
    stream_name = sys.argv[2]
    device_id = sys.argv[3] if len(sys.argv) > 3 else stream_name
    password = sys.argv[4] if len(sys.argv) > 4 else ""

    # Register the device
    register_device(server_url, stream_name, device_id, password)

    # Get server status
    get_server_status(server_url)


if __name__ == "__main__":
    main()
