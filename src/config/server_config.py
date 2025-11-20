"""
Server Configuration
Central configuration for the CrowdCount Counter Server
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
STREAMS_DIR = DATA_DIR / "streams"
RECORDINGS_DIR = DATA_DIR / "recordings"

# MediaMTX settings
MEDIAMTX_BINARY = BASE_DIR / "bin" / "mediamtx"
MEDIAMTX_CONFIG = SRC_DIR / "config" / "mediamtx.yml"
MEDIAMTX_API_URL = "http://127.0.0.1:9997"
MEDIAMTX_METRICS_URL = "http://127.0.0.1:9998/metrics"

# RTSP settings
RTSP_HOST = "127.0.0.1"
RTSP_PORT = 8554

# Orchestrator settings
ORCHESTRATOR_HOST = "0.0.0.0"
ORCHESTRATOR_PORT = 5000

# Stream processor settings
PROCESSOR_TYPE = "subprocess"  # Options: "subprocess", "docker" (docker not implemented yet)
MAX_PROCESSORS = 10  # Maximum number of concurrent stream processors
PROCESSOR_RESTART_DELAY = 5  # Seconds to wait before restarting a failed processor

# Stream settings
STREAM_RECONNECT_TIMEOUT = 30  # Seconds to wait for stream reconnection before stopping processor
STREAM_HEALTH_CHECK_INTERVAL = 10  # Seconds between stream health checks

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = DATA_DIR / "server.log"

# Database settings
DB_PATH = DATA_DIR / "devices.db"

# SSH settings for remote device control
SSH_KEY_PATH = os.path.expanduser('~/.ssh/id_rsa')
SSH_USER = 'pi'  # Default Raspberry Pi username

# Web dashboard
STATIC_DIR = BASE_DIR / "static"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
STREAMS_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
