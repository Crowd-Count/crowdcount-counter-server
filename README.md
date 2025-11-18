# CrowdCount Counter Server

Server-side application for receiving and processing video streams from Raspberry Pi cameras over Tailscale.

## Architecture

```
┌─────────────────┐
│  Raspberry Pi 1 │─┐
└─────────────────┘ │
                    │
┌─────────────────┐ │    ┌──────────────────────────────────────────┐
│  Raspberry Pi 2 │─┼───▶│  This PC (Counter Server)                │
└─────────────────┘ │    │                                          │
                    │    │  ┌────────────┐   ┌──────────────────┐  │
┌─────────────────┐ │    │  │  MediaMTX  │   │   Orchestrator   │  │
│  Raspberry Pi N │─┘    │  │   Server   │◀─▶│   (Manager)      │  │
└─────────────────┘      │  └────────────┘   └──────────────────┘  │
                         │         │                   │            │
    RTSP/Tailscale       │         │                   ▼            │
                         │         │          ┌─────────────────┐   │
                         │         └─────────▶│   Processor 1   │   │
                         │                    ├─────────────────┤   │
                         │                    │   Processor 2   │   │
                         │                    ├─────────────────┤   │
                         │                    │   Processor N   │   │
                         │                    └─────────────────┘   │
                         └──────────────────────────────────────────┘
```

## Components

### 1. MediaMTX Server
- Receives RTSP streams from Raspberry Pis
- Provides HLS and WebRTC endpoints for viewing
- Notifies orchestrator when streams start/stop

### 2. Stream Orchestrator
- Manages MediaMTX server lifecycle
- Monitors for new streams
- Automatically spawns a processor for each stream
- Monitors processor health and restarts if needed
- Provides REST API for status

### 3. Stream Processors
- Each stream gets its own isolated processor (subprocess)
- Pulls frames from MediaMTX via RTSP
- Runs computer vision algorithms
- Base class provided for easy extension

## Directory Structure

```
crowdcount-counter-server/
├── bin/                        # Binary files (MediaMTX)
│   └── mediamtx               # Downloaded by setup.sh
├── data/                       # Runtime data
│   ├── streams/               # Stream metadata
│   ├── recordings/            # Video recordings (if enabled)
│   └── server.log            # Server logs
├── src/
│   ├── config/
│   │   ├── mediamtx.yml      # MediaMTX configuration
│   │   └── server_config.py   # Server settings
│   ├── streaming/
│   │   ├── mediamtx_manager.py      # MediaMTX process manager
│   │   └── stream_orchestrator.py   # Main orchestrator
│   └── processors/
│       ├── base_processor.py        # Base class for processors
│       └── example_processor.py     # Example implementation
├── scripts/
│   ├── setup.sh              # Setup script (downloads MediaMTX)
│   └── run_server.py         # Main entry point
└── README.md
```

## Setup

### 1. Run Setup Script

This will download MediaMTX and check dependencies:

```bash
cd crowdcount-counter-server
./scripts/setup.sh
```

### 2. Install Python Dependencies

If any are missing:

```bash
pip3 install opencv-python flask requests numpy
```

### 3. Verify Tailscale

Make sure Tailscale is running:

```bash
tailscale status
```

Get your PC's Tailscale IP (you'll need this for the Raspberry Pis):

```bash
tailscale ip -4
```

## Usage

### Starting the Server

```bash
cd crowdcount-counter-server
python3 scripts/run_server.py
```

This will:
1. Start MediaMTX on port 8554 (RTSP)
2. Start the orchestrator API on port 5000
3. Monitor for incoming streams
4. Automatically spawn processors for each stream

### Connecting Raspberry Pis

On each Raspberry Pi, run:

```bash
python3 stream_to_mediamtx.py <YOUR_PC_TAILSCALE_IP> <stream_name>
```

Example:
```bash
python3 stream_to_mediamtx.py 100.64.1.2 camera1
```

### Viewing Streams

Once a stream is active, you can view it:

- **RTSP**: `rtsp://<YOUR_TAILSCALE_IP>:8554/<stream_name>`
- **HLS (browser)**: `http://<YOUR_TAILSCALE_IP>:8554/<stream_name>/`
- **VLC**: Open network stream with RTSP URL

### Monitoring

Check server status:
```bash
curl http://localhost:5000/status
```

Check active processors:
```bash
curl http://localhost:5000/processors
```

View MediaMTX stats:
```bash
curl http://localhost:9997/v3/paths/list
```

## Creating Custom Processors

Replace `ExampleProcessor` with your own computer vision logic:

### 1. Create Your Processor Class

```python
# src/processors/crowd_counter_processor.py

from processors.base_processor import BaseStreamProcessor
import numpy as np

class CrowdCounterProcessor(BaseStreamProcessor):
    def on_start(self):
        # Load your ML models here
        self.logger.info("Loading crowd counting model...")
        # self.model = load_model(...)

    def process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float):
        # Your computer vision logic here
        # Example: Detect and count people

        # count = self.model.predict(frame)
        # self.logger.info(f"Detected {count} people")

        # Return processed frame for visualization (optional)
        return frame

    def on_stop(self):
        # Cleanup
        self.logger.info("Stopping crowd counter")
        # Save results, close connections, etc.
```

### 2. Use Your Processor

Edit `scripts/run_server.py`:

```python
from processors.crowd_counter_processor import CrowdCounterProcessor

# Change this line:
orchestrator = StreamOrchestrator(processor_class=CrowdCounterProcessor)
```

### 3. Restart Server

```bash
python3 scripts/run_server.py
```

Now each stream will automatically use your custom processor!

## Configuration

### MediaMTX Configuration

Edit `src/config/mediamtx.yml` to customize:
- Ports
- Authentication
- Recording settings
- Performance tuning

### Server Configuration

Edit `src/config/server_config.py` to customize:
- Max concurrent processors
- Health check intervals
- Logging levels
- Paths

## Processor Isolation

Each stream gets its own subprocess for:
- **Isolation**: Crashes don't affect other streams
- **Performance**: True parallel processing
- **Flexibility**: Different processors can use different resources

## Performance

For 10 streams at 640p 30fps:
- **CPU**: ~50-100% depending on processing (8-core recommended)
- **RAM**: ~500MB - 1GB per processor
- **Network**: ~20-40 Mbps total (2-4 Mbps per stream)
- **Disk**: Minimal (unless recording enabled)

## Troubleshooting

### MediaMTX won't start

```bash
# Check if port 8554 is in use
sudo lsof -i :8554

# Check logs
tail -f data/mediamtx.log
```

### Processor crashes immediately

```bash
# Test RTSP connection manually
ffplay rtsp://127.0.0.1:8554/camera1

# Check processor logs
# They'll appear in the server output
```

### Stream not appearing

```bash
# Check MediaMTX paths
curl http://localhost:9997/v3/paths/list

# Verify Raspberry Pi is connected
tailscale status
ping <RASPBERRY_PI_TAILSCALE_IP>
```

### High CPU usage

- Reduce stream resolution on Raspberry Pi
- Reduce FPS
- Optimize your processor's `process_frame()` method
- Reduce number of concurrent streams

## Advanced Features

### Recording Streams

Edit `src/config/mediamtx.yml`:

```yaml
paths:
  camera1:
    record: yes
    recordFormat: mp4
```

Recordings will be saved to `data/recordings/camera1/`

### Authentication

Edit `src/config/mediamtx.yml`:

```yaml
paths:
  all:
    publishUser: pi
    publishPass: secret_password
```

Update Raspberry Pi command:
```bash
python3 stream_to_mediamtx.py 100.64.1.2 camera1 --user pi --pass secret_password
```
(Note: You'd need to add auth support to the Pi script)

### Docker Deployment (Future)

Coming soon: Docker containers for each processor instead of subprocesses.

## API Reference

### GET /status
Returns server status and stream count

### GET /processors
Returns information about all active processors

### POST /stream/ready
Internal webhook called by MediaMTX when stream starts

### POST /stream/stopped
Internal webhook called by MediaMTX when stream stops

## Next Steps

1. ✅ Set up server
2. ✅ Connect Raspberry Pis
3. **Implement your computer vision processor**
4. Test with multiple streams
5. Deploy and monitor

## Support

For issues or questions, check:
- MediaMTX docs: https://github.com/bluenviron/mediamtx
- This project's issues (if on GitHub)
