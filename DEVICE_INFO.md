# Device Information & Authentication

The CrowdCount Counter Server now supports device identification and optional password authentication for each connected stream.

## Overview

When Raspberry Pi cameras connect to the server, they can provide:
- **Device ID**: A unique identifier for the device (e.g., "crowdcount1", "entrance-camera")
- **Password**: An optional authentication password for device verification

This information is stored for the session and passed to stream processors, enabling:
- Device-specific processing logic
- Authentication and access control (future feature)
- Better logging and tracking
- Device identification in analytics

## How It Works

### 1. Device Registration

Raspberry Pis can register their device information in two ways:

#### Method A: Pre-registration (Recommended)
Register device info before the stream connects:

```bash
curl -X POST http://SERVER_IP:5000/device/register \
  -H "Content-Type: application/json" \
  -d '{
    "stream_name": "crowdcount1",
    "device_id": "entrance-camera-01",
    "password": "your-secure-password"
  }'
```

#### Method B: Include in Stream Ready Notification
When MediaMTX notifies the server of a new stream, it can include device info:

```json
POST /stream/ready
{
  "stream": "crowdcount1",
  "device_id": "entrance-camera-01",
  "password": "your-secure-password"
}
```

### 2. Session Storage

Device information is stored in memory for the duration of the stream session:

```python
{
  "device_id": "entrance-camera-01",
  "password": "your-secure-password",
  "timestamp": 1699999999.123
}
```

### 3. Passed to Processors

When a processor is created for a stream, the device info is automatically provided:

```python
class YourProcessor(BaseStreamProcessor):
    def __init__(self, stream_name: str, rtsp_url: str, device_info: dict = None):
        super().__init__(stream_name, rtsp_url, device_info)
        # Access device info:
        # self.device_id - The device ID (defaults to stream_name if not provided)
        # self.password - The password (empty string if not provided)
        # self.device_info - Full dict with all device information
```

## API Endpoints

### Register Device
**POST** `/device/register`

Register or update device information for a stream.

**Request:**
```json
{
  "stream_name": "crowdcount1",
  "device_id": "entrance-camera-01",
  "password": "optional-password"
}
```

**Response:**
```json
{
  "status": "ok",
  "stream_name": "crowdcount1",
  "device_id": "entrance-camera-01"
}
```

### Get Status (includes device info)
**GET** `/status`

**Response:**
```json
{
  "running": true,
  "mediamtx_healthy": true,
  "active_streams": ["crowdcount1", "crowdcount2"],
  "processor_count": 2,
  "max_processors": 10,
  "device_info": {
    "crowdcount1": {
      "device_id": "entrance-camera-01",
      "has_password": true
    },
    "crowdcount2": {
      "device_id": "crowdcount2",
      "has_password": false
    }
  }
}
```

## Usage in Processors

### Basic Usage

```python
class CrowdCountProcessor(BaseStreamProcessor):
    def on_start(self):
        # Log device information
        self.logger.info(f"Processing stream from device: {self.device_id}")

        # Check if device has password
        if self.password:
            self.logger.info("Device authenticated with password")

        # Access full device info
        timestamp = self.device_info.get('timestamp', 0)
```

### Device-Specific Processing

```python
class CrowdCountProcessor(BaseStreamProcessor):
    def on_start(self):
        # Load device-specific configuration
        if self.device_id == "entrance-camera-01":
            self.counting_line_y = 300
            self.direction = "up"
        elif self.device_id == "exit-camera-01":
            self.counting_line_y = 400
            self.direction = "down"
```

### Future: Authentication

```python
class SecureProcessor(BaseStreamProcessor):
    def on_start(self):
        # Example: Verify password against database
        if not self.verify_device_password(self.device_id, self.password):
            raise Exception("Invalid device credentials")

        self.logger.info(f"Device {self.device_id} authenticated successfully")
```

## Configuration in device-info.conf

On the Raspberry Pi side, you can configure device info in `device-info.conf`:

```ini
# Device Configuration
DEVICE_ID=entrance-camera-01
PASSWORD=your-secure-password
SERVER_IP=100.93.86.117
```

The streaming script will automatically send this information when connecting.

## Security Notes

1. **Passwords are stored in memory only** - They are not persisted to disk
2. **Passwords are cleaned up** when streams disconnect
3. **HTTPS/TLS is recommended** for production deployments to encrypt password transmission
4. **The current implementation stores passwords for processing** but doesn't enforce authentication yet
5. **Future enhancement**: Add password validation and access control

## Example Workflow

1. **Configure device** on Raspberry Pi:
   ```bash
   nano device-info.conf
   # Set DEVICE_ID=entrance-camera-01
   # Set PASSWORD=secure123
   ```

2. **Start stream** from Raspberry Pi:
   ```bash
   ./start_stream.sh
   ```

3. **Server receives** device info and stores it

4. **Processor accesses** device info:
   ```python
   def on_start(self):
       self.logger.info(f"Device: {self.device_id}")
       # Use device_id and password for processing logic
   ```

5. **Stream disconnects** - device info automatically cleaned up

## Troubleshooting

**Device info not showing up:**
- Check that the Raspberry Pi is sending the device_id in the stream notification
- Use `GET /status` to verify device info is stored
- Check server logs for registration messages

**Password not being passed:**
- Verify the password is set in device-info.conf
- Ensure the streaming script includes password in the POST request
- Check that the password field is not empty in the request

**Device ID defaulting to stream name:**
- This is expected if no device_id is provided
- The device_id will default to the stream_name parameter
