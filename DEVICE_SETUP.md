# Device Setup Guide

## Device Naming Convention

This system supports two types of counters:

### Normal Counters (Default)
- **Format**: `device-name` (no @ symbol)
- **SSH Username**: `counter` (always)
- **SSH Hostname**: Same as device_id
- **Authentication**: SSH key (`~/.ssh/id_rsa`)
- **Examples**: `T0002`, `T0003`, `entrance-camera`, `exit-counter`

### Developer Counter (Special Case)
- **Format**: `username@hostname`
- **SSH Username**: Extracted from device_id (before @)
- **Authentication**: Password (prompted on dashboard)
- **Example**: `crowdcount1@crowdcount1` → username: `crowdcount1`

## Device Registration

### Your Current Setup
- Stream `T0001` = Device `crowdcount1@crowdcount1` (developer counter, password auth)
- Stream `T0002+` = Device uses stream name (normal counter, SSH key auth)

### Registration Options

**Option 1: Auto-register (Default)**
- Most devices auto-register when they connect
- Device ID = Stream Name (e.g., T0002 → device_id: T0002)
- Uses SSH key authentication with username `counter`

**Option 2: Pre-register T0001**
- While server is running:
```bash
cd crowdcount-counter-server
./register_devices.sh
```

**Option 3: Manual API Registration**
```bash
# For developer counter (T0001)
curl -X POST http://localhost:5000/device/register \
  -H "Content-Type: application/json" \
  -d '{
    "stream_name": "T0001",
    "device_id": "crowdcount1@crowdcount1"
  }'

# For normal counters (not usually needed - auto-registers)
curl -X POST http://localhost:5000/device/register \
  -H "Content-Type: application/json" \
  -d '{
    "stream_name": "T0002",
    "device_id": "T0002"
  }'
```

## Testing Password Prompt

1. Ensure device is registered as developer counter (has @ in device_id)
2. Open dashboard: http://localhost:5000
3. Find the device (e.g., `crowdcount1@crowdcount1`)
4. Click any action button (Start/Stop/Restart/Logs/Config)
5. **Password modal should appear automatically**
6. Enter SSH password
7. Action completes successfully
8. Subsequent actions won't require password (stored for session)

## Troubleshooting

### Device shows as offline
- Check server logs for connection messages
- Verify device is actively streaming
- Wait 10 seconds and refresh dashboard (auto-refresh)

### Password prompt doesn't appear
- Check device_id has @ symbol: `SELECT device_id FROM devices;`
- Open browser console (F12) and check for errors
- Verify response status is 401 (not 500)

### Authentication failed
- For developer counters: Enter correct SSH password in prompt
- For normal counters: Ensure SSH key is set up correctly

## Quick Reference

| Stream Name | Device ID | SSH User | Auth Method | Notes |
|------------|-----------|----------|-------------|-------|
| T0001 | crowdcount1@crowdcount1 | crowdcount1 | Password | Developer counter |
| T0002+ | T0002 (auto) | counter | SSH Key | Normal counters |

## SSH Connection Details

### For T0001 (Developer Counter)
- **Device ID**: `crowdcount1@crowdcount1`
- **SSH Command**: `ssh crowdcount1@<tailscale_ip>` (with password)
- **Dashboard**: Prompts for password when needed

### For Other Devices (Normal Counters)
- **Device ID**: Same as stream name (e.g., `T0002`)
- **SSH Command**: `ssh counter@<tailscale_ip>` (with SSH key)
- **Dashboard**: Works automatically if SSH key is set up

## Adding New Normal Counters

No registration needed! When a new device connects:
1. Stream connects with name (e.g., `T0003`)
2. Device auto-registers as `T0003`
3. SSH uses username `counter` with key auth
4. Ready to manage from dashboard!
