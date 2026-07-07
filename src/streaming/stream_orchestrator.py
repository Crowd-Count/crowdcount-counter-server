"""
Stream Orchestrator
Main orchestrator that manages MediaMTX and stream processors
Automatically spawns a processor for each new stream
"""

import os
import sys
import re
import time
import json
import logging
import threading
import signal
from pathlib import Path
from functools import wraps
from typing import Dict, Type, Optional
from flask import (
    Flask,
    request,
    jsonify,
    Response,
    render_template_string,
    send_from_directory,
)
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import multiprocessing
import requests as req_lib
import cv2
import queue
import sqlite3
import paramiko
import subprocess
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from contextlib import contextmanager

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.crowdcount_api_client import CrowdCountAPIClient
from api.count_backlog import CountBacklog
from config.server_config import (
    ORCHESTRATOR_HOST,
    ORCHESTRATOR_PORT,
    MAX_PROCESSORS,
    STREAM_HEALTH_CHECK_INTERVAL,
    STREAM_RECONNECT_TIMEOUT,
    RTSP_HOST,
    RTSP_PORT,
    LOG_LEVEL,
    DB_PATH,
    SSH_KEY_PATH,
    SSH_USER,
    STATIC_DIR,
    VIDEO_FPS,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    NUM_GPUS,
    GPU_SLOT_LIMITS,
    GPU_PRIORITY_ORDER,
    INFERENCE_QUEUE_MAXSIZE,
    INFERENCE_IMGSZ,
    INFERENCE_BATCH_SIZE,
    INFERENCE_BATCH_WAIT_MS,
    TENSORRT_ENABLED,
    TENSORRT_ENGINE_DIR,
    TENSORRT_MAX_BATCH,
    TENSORRT_WARMUP_ITERS,
    BACKLOG_DB_PATH,
    BACKLOG_FLUSH_INTERVAL,
    BACKLOG_FLUSH_BATCH_SIZE,
)
from streaming.mediamtx_manager import MediaMTXManager
from processors import BaseStreamProcessor, ExampleProcessor
from processors.inference_server import InferenceServer, RequestType, InferenceRequest


# Database helper functions
def init_db():
    """Create database tables if they don't exist"""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            stream_name TEXT NOT NULL,
            tailscale_ip TEXT,
            last_seen TIMESTAMP,
            status TEXT DEFAULT 'offline',
            width INTEGER DEFAULT {VIDEO_WIDTH},
            height INTEGER DEFAULT {VIDEO_HEIGHT},
            fps INTEGER DEFAULT {VIDEO_FPS},
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# Thread-local storage for database connections
_db_local = threading.local()


def _get_thread_db():
    """Get or create a database connection for the current thread."""
    if not hasattr(_db_local, "connection") or _db_local.connection is None:
        _db_local.connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _db_local.connection.row_factory = sqlite3.Row
    return _db_local.connection


@contextmanager
def get_db():
    """Context manager for database access with thread-local connection pooling."""
    conn = _get_thread_db()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise


def close_db_connections():
    """Close thread-local database connections."""
    if hasattr(_db_local, "connection") and _db_local.connection:
        _db_local.connection.close()
        _db_local.connection = None


def _cleanup_queue(q):
    """Safely cleanup a multiprocessing queue."""
    if q is None:
        return
    try:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        q.close()
        joiner = threading.Thread(target=q.join_thread, daemon=True)
        joiner.start()
        joiner.join(timeout=5)
    except Exception:
        pass


def _get_fd_count():
    """Get current open file descriptor count."""
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except Exception:
        return -1


def _load_locations() -> dict:
    """Load device_id -> location_name mapping from config/locations.json.

    Returns an empty dict on file-not-found or JSON errors so callers
    always get a safe value.  The file is read fresh on every call so
    edits take effect without a restart.
    """
    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "locations.json"
        with open(config_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _load_mirror_streams() -> dict:
    """Load mirror stream config from config/mirror_streams.json.

    Returns a dict mapping mirror stream names to their config,
    or empty dict on errors.
    """
    try:
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "mirror_streams.json"
        )
        with open(config_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def ssh_exec(
    ip: str, command: str, device_id: str = None, password: str = None
) -> tuple:
    """
    Execute command via SSH on remote device

    Args:
        ip: Tailscale IP address of the device
        command: Command to execute
        device_id: Device ID (used to determine username for developer counters)
        password: SSH password (optional, for developer counters)

    Returns:
        tuple: (success: bool, result: str)
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Determine username based on device_id
        # Developer counters: format is "username@hostname" (e.g., "crowdcount1@crowdcount1")
        # Normal counters: use username "counter"
        username = SSH_USER  # Default: 'pi'

        if device_id:
            if "@" in device_id:
                # Developer counter: extract username from device_id
                username = device_id.split("@")[0]
            else:
                # Normal counter: use "counter" as username
                username = "counter"

        # Try password authentication if provided
        if password:
            ssh.connect(
                ip,
                username=username,
                password=password,
                timeout=10,
                allow_agent=False,
                look_for_keys=False,
            )
        else:
            # Try key-based authentication (for normal counters or if no password)
            ssh.connect(
                ip,
                username=username,
                key_filename=SSH_KEY_PATH,
                timeout=10,
                allow_agent=True,
                look_for_keys=True,
            )

        stdin, stdout, stderr = ssh.exec_command(command)
        output = stdout.read().decode("utf-8")
        error = stderr.read().decode("utf-8")
        exit_code = stdout.channel.recv_exit_status()

        ssh.close()

        success = exit_code == 0
        result = output if success else error
        return success, result

    except paramiko.AuthenticationException as e:
        return False, f"Authentication failed: {str(e)}"
    except Exception as e:
        return False, str(e)


def update_device_status():
    """Update status of all devices based on last_seen"""
    with get_db() as conn:
        c = conn.cursor()

        # Mark devices as offline if not seen in 60 seconds
        threshold = datetime.now() - timedelta(seconds=60)
        c.execute(
            """
            UPDATE devices
            SET status = 'offline'
            WHERE last_seen < ? AND status != 'offline'
        """,
            (threshold,),
        )

        conn.commit()


class StreamOrchestrator:
    """
    Orchestrates stream reception and processing
    - Manages MediaMTX server
    - Monitors for new streams
    - Spawns processors for each stream
    - Monitors processor health
    """

    def __init__(self, processor_class: Type[BaseStreamProcessor] = ExampleProcessor):
        """
        Initialize the orchestrator

        Args:
            processor_class: The processor class to use for each stream (default: ExampleProcessor)
        """
        self.mediamtx = MediaMTXManager()
        self.processor_class = processor_class

        # Track active processors
        self.processors: Dict[str, multiprocessing.Process] = {}

        # Track frame queues for live viewing
        self.frame_queues: Dict[str, multiprocessing.Queue] = {}

        # Track API clients for device status updates (online/offline)
        self.api_clients: Dict[str, CrowdCountAPIClient] = {}

        # Track device information for each stream (device_id, password, etc.)
        self.device_info: Dict[str, dict] = {}

        # Track SSH passwords for devices (for developer counters)
        # Format: {device_id: password}
        self.ssh_passwords: Dict[str, str] = {}

        # Track streams pending reconnection (grace period before marking offline)
        # Format: {stream_name: {'stop_time': float, 'api_client': CrowdCountAPIClient}}
        self.pending_reconnect: Dict[str, dict] = {}

        # Track consecutive processor restart failures per stream
        # Prevents infinite restart loops when stream is registered but not actually sending data
        # Format: {stream_name: {'count': int, 'last_failure': float}}
        self._restart_failures: Dict[str, dict] = {}

        # Track FPS stats per stream for viewer display
        # Format: {stream_name: {'frame_count': int, 'last_reset': float, 'fps': float}}
        self.stream_stats: Dict[str, dict] = {}

        # Track streams currently being started (to prevent duplicate processor creation)
        # Maps stream_name -> timestamp when start began (for stale entry cleanup)
        self._starting_processors: Dict[str, float] = {}

        # Track when each processor was started (for uptime display)
        self._processor_start_times: Dict[str, float] = {}

        # Centralized inference server state
        self.inference_servers: Dict[
            int, multiprocessing.Process
        ] = {}  # gpu_id -> Process
        self.inference_request_queues: Dict[
            int, multiprocessing.Queue
        ] = {}  # gpu_id -> shared input queue
        # Pre-allocated result queue pools per GPU: {gpu_id: {slot_name: Queue}}
        self._result_queue_pools: Dict[int, Dict[str, multiprocessing.Queue]] = {}
        # Track which slots are in use: {gpu_id: {slot_name: camera_id or None}}
        self._slot_assignments: Dict[int, Dict[str, Optional[str]]] = {}
        # Camera -> (gpu_id, slot_name) mapping
        self.camera_gpu_assignments: Dict[str, tuple] = {}
        # Heartbeat values for inference server liveness monitoring
        self._inference_heartbeats: Dict[int, multiprocessing.Value] = {}

        # Heartbeat values for per-processor stuck detection
        self._processor_heartbeats: Dict[str, multiprocessing.Value] = {}

        self.running = False
        self.monitor_thread = None
        self._backlog = None  # CountBacklog instance, initialized in start()

        # Monitor thread watchdog: heartbeat stamped each monitor loop iteration
        self._monitor_heartbeat = 0.0
        self._monitor_heartbeat_lock = threading.Lock()

        # Thread pool for fire-and-forget status updates (prevents thread accumulation)
        self._status_update_executor = ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="StatusUpdate"
        )
        self._stopping = False  # Guard against signal handler re-entry

        # Mirror streams: set of stream names that are mirrors of prod streams
        # Loaded at startup from config/mirror_streams.json
        self._mirror_streams: set = set()

        # Thread safety lock for shared state (RLock allows nested acquisition)
        self._state_lock = threading.RLock()

        # Setup logging
        self.logger = logging.getLogger("Orchestrator")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, LOG_LEVEL))

        # Initialize database
        init_db()

        # Load mirror stream config and pre-populate device_info
        mirror_config = _load_mirror_streams()
        for stream_name, config in mirror_config.items():
            self._mirror_streams.add(stream_name)
            self.device_info[stream_name] = {
                "device_id": config["device_id"],
                "password": "",  # Empty = API client disabled (no counting calls)
                "timestamp": time.time(),
                "is_mirror": True,
            }
        if self._mirror_streams:
            self.logger.info(
                f"Loaded {len(self._mirror_streams)} mirror streams: {sorted(self._mirror_streams)}"
            )

        # Flask app for receiving MediaMTX webhooks and device management
        self.app = Flask(__name__, static_folder=str(STATIC_DIR))
        CORS(self.app)
        self.limiter = Limiter(
            get_remote_address,
            app=self.app,
            default_limits=["60 per minute"],
            storage_uri="memory://",
        )
        self.setup_routes()

    @staticmethod
    def _is_valid_identifier(value: str, max_length: int = 128) -> bool:
        """Validate that a string is a safe identifier (alphanumeric, hyphens, underscores, @, dots)."""
        if not value or len(value) > max_length:
            return False
        return bool(re.match(r"^[a-zA-Z0-9@._-]+$", value))

    def _validate_device_id(self, f):
        """Decorator to validate device_id path parameter."""

        @wraps(f)
        def decorated(*args, **kwargs):
            device_id = kwargs.get("device_id", "")
            if not self._is_valid_identifier(device_id):
                return jsonify({"error": "Invalid device ID"}), 400
            return f(*args, **kwargs)

        return decorated

    def setup_routes(self):
        """Setup Flask routes for MediaMTX webhooks"""

        @self.app.route("/stream/ready", methods=["POST"])
        @self.limiter.limit("50000 per minute")
        def stream_ready():
            """Called by MediaMTX when a stream becomes ready"""
            data = request.get_json()
            if not data:
                return jsonify(
                    {"status": "error", "message": "Invalid request body"}
                ), 400
            stream_name = data.get("stream", "")

            if stream_name and self._is_valid_identifier(stream_name):
                # Cancel any pending reconnect timeout (stream came back)
                with self._state_lock:
                    if stream_name in self.pending_reconnect:
                        elapsed = (
                            time.time()
                            - self.pending_reconnect[stream_name]["stop_time"]
                        )
                        self.logger.info(
                            f"Stream {stream_name} reconnected after {elapsed:.1f}s - canceling offline timeout"
                        )
                        del self.pending_reconnect[stream_name]

                    # If a processor already exists and is alive, stop it first.
                    # runOnReady means MediaMTX has a new publisher - the old processor
                    # is likely stuck reading from the dead previous connection.
                    needs_restart = (
                        stream_name in self.processors
                        and self.processors[stream_name].is_alive()
                    )

                if needs_restart:
                    self.logger.info(
                        f"Stream {stream_name}: new publisher ready but old processor alive - restarting processor"
                    )
                    self.stop_processor(stream_name, reason="stream_ready_restart")

                # Get client IP (Tailscale IP)
                client_ip = request.remote_addr

                # Get device info from pre-registration - reject unregistered streams
                is_mirror = stream_name in self._mirror_streams
                with self._state_lock:
                    if stream_name in self.device_info:
                        device_id = self.device_info[stream_name].get(
                            "device_id", stream_name
                        )
                        password = self.device_info[stream_name].get("password", "")
                    elif is_mirror:
                        # Mirror streams don't need credentials
                        device_id = stream_name
                        password = ""
                        self.device_info[stream_name] = {
                            "device_id": device_id,
                            "password": password,
                            "timestamp": time.time(),
                        }
                    else:
                        # No pre-registration - reject the stream
                        self.logger.warning(
                            f"✗ Rejected unregistered stream: {stream_name} from {request.remote_addr}"
                        )
                        return jsonify(
                            {"status": "error", "message": "Device not registered"}
                        ), 401

                if not is_mirror:
                    # Update device status in database
                    try:
                        with get_db() as conn:
                            c = conn.cursor()
                            c.execute(
                                "SELECT * FROM devices WHERE device_id = ?",
                                (device_id,),
                            )
                            existing = c.fetchone()

                            if existing:
                                # Update existing device
                                c.execute(
                                    """
                                    UPDATE devices
                                    SET stream_name = ?, tailscale_ip = ?, last_seen = ?, status = 'online'
                                    WHERE device_id = ?
                                """,
                                    (stream_name, client_ip, datetime.now(), device_id),
                                )
                            else:
                                # Insert new device
                                c.execute(
                                    """
                                    INSERT INTO devices (device_id, stream_name, tailscale_ip, last_seen, status)
                                    VALUES (?, ?, ?, ?, 'online')
                                """,
                                    (device_id, stream_name, client_ip, datetime.now()),
                                )

                            conn.commit()
                    except Exception as e:
                        self.logger.error(
                            f"Error updating device status in database: {e}"
                        )

                # Detailed logging for testing
                self.logger.info("=" * 60)
                self.logger.info(f"{'MIRROR' if is_mirror else ''} DEVICE CONNECTED")
                self.logger.info(f"   Stream Name: {stream_name}")
                self.logger.info(f"   Device ID:   {device_id}")
                self.logger.info(f"   IP Address:  {client_ip}")
                self.logger.info(f"   Password:    {'***' if password else '(none)'}")
                self.logger.info("=" * 60)

                self.start_processor(stream_name)
                return jsonify({"status": "ok"}), 200

            return jsonify(
                {"status": "error", "message": "No stream name provided"}
            ), 400

        @self.app.route("/stream/stopped", methods=["POST"])
        @self.limiter.limit("50000 per minute")
        def stream_stopped():
            """Called by MediaMTX when a stream stops - uses grace period before marking offline"""
            data = request.get_json()
            if not data:
                return jsonify(
                    {"status": "error", "message": "Invalid request body"}
                ), 400
            stream_name = data.get("stream", "")

            if stream_name and self._is_valid_identifier(stream_name):
                self.logger.info(f"Received stream stopped notification: {stream_name}")

                # Stop the processor but don't mark offline yet - use grace period
                self._handle_stream_interruption(stream_name)

                return jsonify({"status": "ok"}), 200

            return jsonify(
                {"status": "error", "message": "No stream name provided"}
            ), 400

        @self.app.route("/auth", methods=["POST"])
        @self.limiter.limit("50000 per minute")
        def auth_device():
            """
            Authentication endpoint for MediaMTX
            Validates device credentials against CrowdCount API

            Expected MediaMTX format:
            {
                "user": "device_id",
                "password": "api_password",
                "ip": "client_ip",
                "action": "publish" or "read" or "api"
            }
            """
            data = request.get_json() or {}
            user = data.get("user", "")
            password = data.get("password", "")
            action = data.get("action", "")

            if not user or not password:
                self.logger.warning(
                    f"✗ Auth attempt with missing credentials from {request.remote_addr}"
                )
                return "", 401  # Unauthorized

            # Validate against CrowdCount API
            is_valid = CrowdCountAPIClient.check_password(user, password)

            if is_valid:
                self.logger.info(f"✓ Device authenticated: {user} (action: {action})")
                return "", 200  # OK
            else:
                self.logger.warning(
                    f"✗ Authentication failed for {user}: Invalid credentials"
                )
                return "", 401  # Unauthorized

        @self.app.route("/status", methods=["GET"])
        def status():
            """Get orchestrator status - non-blocking to prevent website hangs"""
            # Use timeout to prevent blocking forever during high contention
            # (e.g., mass disconnections causing many webhook handlers to hold the lock)
            acquired = self._state_lock.acquire(timeout=2)
            if not acquired:
                # Lock unavailable - return 503 so website shows error instead of hanging
                self.logger.warning("Status endpoint: lock unavailable (server busy)")
                return jsonify(
                    {
                        "error": "Server busy",
                        "running": self.running,
                        "active_streams": [],
                        "processor_count": 0,
                        "max_processors": MAX_PROCESSORS,
                    }
                ), 503
            try:
                return jsonify(
                    {
                        "running": self.running,
                        "mediamtx_healthy": self.mediamtx.is_healthy(),
                        "active_streams": list(self.processors.keys()),
                        "processor_count": len(self.processors),
                        "max_processors": MAX_PROCESSORS,
                        "device_info": {
                            stream: {
                                "device_id": info["device_id"],
                                "has_password": bool(info.get("password")),
                            }
                            for stream, info in self.device_info.items()
                        },
                    }
                ), 200
            finally:
                self._state_lock.release()

        @self.app.route("/processors", methods=["GET"])
        def processors():
            """Get processor information"""
            with self._state_lock:
                processors_copy = list(self.processors.items())
            processor_info = {}
            for stream_name, proc in processors_copy:
                processor_info[stream_name] = {
                    "alive": proc.is_alive(),
                    "pid": proc.pid,
                }
            return jsonify(processor_info), 200

        @self.app.route("/api/resources", methods=["GET"])
        def get_resources():
            """Get resource usage statistics for monitoring"""
            with self._state_lock:
                processor_count = len(self.processors)
            return jsonify(
                {
                    "open_fds": _get_fd_count(),
                    "thread_count": threading.active_count(),
                    "processor_count": processor_count,
                }
            ), 200

        @self.app.route("/device/register", methods=["POST"])
        @self.limiter.limit("50000 per minute")
        def register_device():
            """
            Register device information for a stream.
            Validates device credentials before allowing registration.
            """
            data = request.get_json()
            if not data:
                return jsonify(
                    {"status": "error", "message": "Invalid request body"}
                ), 400
            stream_name = data.get("stream_name", "")
            device_id = data.get("device_id", stream_name)
            password = data.get("password", "")

            if not stream_name or not self._is_valid_identifier(stream_name):
                return jsonify(
                    {"status": "error", "message": "stream_name is required"}
                ), 400

            if not device_id or not self._is_valid_identifier(device_id):
                return jsonify({"status": "error", "message": "Invalid device_id"}), 400

            if not password:
                self.logger.warning(
                    f"✗ Registration attempt with missing credentials from {request.remote_addr}"
                )
                return jsonify(
                    {"status": "error", "message": "password is required"}
                ), 400

            # Validate credentials against CrowdCount API
            self.logger.info(f"Validating credentials for {device_id}...")
            is_valid = CrowdCountAPIClient.check_password(device_id, password)
            if not is_valid:
                self.logger.warning(
                    f"✗ Registration failed for {device_id}: Invalid credentials"
                )
                return jsonify(
                    {"status": "error", "message": "Invalid credentials"}
                ), 401
            self.logger.info(f"✓ Credentials valid for {device_id}")

            # Clean up stale state from previous connection
            # This handles unclean disconnects where the old RTSP session lingers
            if stream_name in self.processors:
                self.logger.info(
                    f"Stopping stale processor for {stream_name} during re-registration"
                )
                self.stop_processor(stream_name, reason="register_device_cleanup")

            kicked = self.mediamtx.kick_path_publishers(stream_name)
            if kicked:
                self.logger.info(
                    f"Kicked {kicked} stale publisher session(s) for {stream_name}"
                )

            with self._state_lock:
                self.pending_reconnect.pop(stream_name, None)

            if kicked:
                time.sleep(
                    1
                )  # Let MediaMTX fully clean up before counter starts FFmpeg

            # Get client IP (Tailscale IP)
            client_ip = request.remote_addr

            # Store device info in memory
            with self._state_lock:
                self.device_info[stream_name] = {
                    "device_id": device_id,
                    "password": password,
                    "timestamp": time.time(),
                }

            # Store device info in database
            try:
                with get_db() as conn:
                    c = conn.cursor()

                    # Check if device exists
                    c.execute("SELECT * FROM devices WHERE device_id = ?", (device_id,))
                    existing = c.fetchone()

                    if existing:
                        # Update existing device
                        c.execute(
                            """
                            UPDATE devices
                            SET stream_name = ?, tailscale_ip = ?, last_seen = ?, status = 'online'
                            WHERE device_id = ?
                        """,
                            (stream_name, client_ip, datetime.now(), device_id),
                        )
                    else:
                        # Insert new device
                        c.execute(
                            """
                            INSERT INTO devices (device_id, stream_name, tailscale_ip, last_seen, status)
                            VALUES (?, ?, ?, ?, 'online')
                        """,
                            (device_id, stream_name, client_ip, datetime.now()),
                        )

                    conn.commit()

                self.logger.info(
                    f"Registered device: {device_id} (stream: {stream_name}, IP: {client_ip})"
                )

                # Reset restart failure tracking - device has re-registered fresh
                with self._state_lock:
                    self._restart_failures.pop(stream_name, None)

                return jsonify(
                    {"status": "ok", "stream_name": stream_name, "device_id": device_id}
                ), 200

            except Exception as e:
                self.logger.error(f"Error registering device in database: {e}")
                # Still return success for backward compatibility
                return jsonify(
                    {"status": "ok", "stream_name": stream_name, "device_id": device_id}
                ), 200

        @self.app.route("/view/<stream_name>")
        def view_stream(stream_name):
            """Serve HTML viewer for a stream"""
            if not self._is_valid_identifier(stream_name):
                return jsonify({"error": "Invalid stream name"}), 400
            locations = _load_locations()
            with self._state_lock:
                device_info = self.device_info.get(stream_name, {})
            device_id = device_info.get("device_id", stream_name)
            location_name = locations.get(device_id, "")

            html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{{ location_name or stream_name }} - CrowdCount DEV</title>
                <style>
                    :root {
                        --bg-primary: #0a0a0a;
                        --bg-secondary: #141414;
                        --bg-card: #1a1a1a;
                        --text-primary: #ffffff;
                        --text-secondary: #888888;
                        --accent: #3b82f6;
                        --accent-hover: #2563eb;
                        --success: #22c55e;
                        --warning: #f59e0b;
                        --error: #ef4444;
                        --border: #2a2a2a;
                    }
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        background: var(--bg-primary);
                        color: var(--text-primary);
                        min-height: 100vh;
                    }
                    .env-banner {
                        position: fixed;
                        top: 0;
                        left: 0;
                        right: 0;
                        padding: 6px;
                        text-align: center;
                        font-size: 11px;
                        font-weight: 700;
                        letter-spacing: 2px;
                        z-index: 1000;
                        background: var(--warning);
                        color: #000;
                    }
                    .app {
                        max-width: 1400px;
                        margin: 0 auto;
                        padding: 50px 24px 24px;
                    }
                    .header {
                        text-align: center;
                        padding: 20px 0;
                    }
                    .header h1 {
                        font-size: 1.5rem;
                        font-weight: 600;
                        margin-bottom: 4px;
                    }
                    .header .stream-id {
                        color: var(--text-secondary);
                        font-size: 14px;
                    }
                    .controls {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        gap: 16px;
                        padding: 16px;
                        background: var(--bg-secondary);
                        border-radius: 12px;
                        margin-bottom: 20px;
                        border: 1px solid var(--border);
                        flex-wrap: wrap;
                    }
                    .btn {
                        display: inline-flex;
                        align-items: center;
                        gap: 8px;
                        padding: 10px 20px;
                        font-size: 14px;
                        font-weight: 500;
                        border: none;
                        border-radius: 8px;
                        cursor: pointer;
                        transition: all 0.2s ease;
                    }
                    .btn.active {
                        background: var(--success);
                        color: #000;
                    }
                    .status-badge {
                        display: inline-flex;
                        align-items: center;
                        gap: 6px;
                        padding: 6px 12px;
                        border-radius: 20px;
                        font-size: 12px;
                        font-weight: 600;
                    }
                    .status-badge.overlay {
                        background: rgba(34, 197, 94, 0.15);
                        color: var(--success);
                    }
                    .status-badge.clean {
                        background: rgba(59, 130, 246, 0.15);
                        color: var(--accent);
                    }
                    .status-badge.fps {
                        background: rgba(255, 255, 255, 0.1);
                        color: var(--text-primary);
                        font-family: 'SF Mono', Monaco, monospace;
                    }
                    .status-badge.healthy {
                        background: rgba(34, 197, 94, 0.15);
                        color: var(--success);
                    }
                    .status-badge.warning {
                        background: rgba(245, 158, 11, 0.15);
                        color: var(--warning);
                    }
                    .status-badge.error {
                        background: rgba(239, 68, 68, 0.15);
                        color: var(--error);
                    }
                    .status-dot {
                        width: 8px;
                        height: 8px;
                        border-radius: 50%;
                        background: currentColor;
                    }
                    .video-container {
                        background: var(--bg-card);
                        border: 1px solid var(--border);
                        border-radius: 16px;
                        overflow: hidden;
                        position: relative;
                    }
                    .video-container img {
                        width: 100%;
                        display: block;
                    }
                    .footer {
                        text-align: center;
                        padding: 20px;
                        color: var(--text-secondary);
                        font-size: 13px;
                    }
                </style>
            </head>
            <body>
                <div class="env-banner">DEVELOPMENT</div>
                <div class="app">
                    <header class="header">
                        <h1>{{ location_name or 'Stream Viewer' }}</h1>
                        <p class="stream-id">{{ device_id }}{% if device_id != stream_name %} &middot; {{ stream_name }}{% endif %}</p>
                    </header>
                    <div class="controls">
                        <button id="toggleBtn" class="btn active" onclick="toggleOverlay()">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="3" y="3" width="18" height="18" rx="2"/>
                                <circle cx="8.5" cy="8.5" r="1.5"/>
                                <path d="M21 15l-5-5L5 21"/>
                            </svg>
                            Overlay: ON
                        </button>
                        <span id="statusBadge" class="status-badge overlay">
                            <span class="status-dot"></span>
                            Tracking Enabled
                        </span>
                        <span id="inputFpsBadge" class="status-badge fps">IN: -- FPS</span>
                        <span id="streamFpsBadge" class="status-badge fps">OUT: -- FPS</span>
                        <span id="resolutionBadge" class="status-badge fps" style="display:none"></span>
                        <span id="uptimeBadge" class="status-badge fps" style="display:none"></span>
                        <span id="healthBadge" class="status-badge healthy">
                            <span class="status-dot"></span>
                            --
                        </span>
                    </div>
                    <div class="video-container">
                        <img id="streamImg" src="/video_feed/{{ stream_name }}?overlay=true" alt="Live Stream">
                    </div>
                    <div class="footer">
                        <p id="feedInfo">Live Detection Feed</p>
                    </div>
                </div>
                <script>
                    let overlayEnabled = true;
                    const streamName = '{{ stream_name }}';

                    function formatUptime(seconds) {
                        if (!seconds || seconds <= 0) return '--';
                        const h = Math.floor(seconds / 3600);
                        const m = Math.floor((seconds % 3600) / 60);
                        const s = seconds % 60;
                        if (h > 0) return h + 'h ' + m + 'm';
                        if (m > 0) return m + 'm ' + s + 's';
                        return s + 's';
                    }

                    async function updateFps() {
                        try {
                            const response = await fetch('/api/stream/' + streamName + '/stats');
                            const data = await response.json();
                            document.getElementById('inputFpsBadge').textContent = 'IN: ' + data.input_fps.toFixed(1) + ' FPS';
                            document.getElementById('streamFpsBadge').textContent = 'OUT: ' + data.stream_fps.toFixed(1) + ' FPS';

                            // Resolution badge
                            const resBadge = document.getElementById('resolutionBadge');
                            if (data.resolution) {
                                resBadge.textContent = data.resolution;
                                resBadge.style.display = '';
                            }

                            // Uptime badge
                            const uptBadge = document.getElementById('uptimeBadge');
                            if (data.uptime_seconds > 0) {
                                uptBadge.textContent = 'Up: ' + formatUptime(data.uptime_seconds);
                                uptBadge.style.display = '';
                            }

                            // Health badge
                            const healthBadge = document.getElementById('healthBadge');
                            const fps = data.input_fps || 0;
                            if (!data.active || fps === 0) {
                                healthBadge.innerHTML = '<span class="status-dot"></span> Offline';
                                healthBadge.className = 'status-badge error';
                            } else if (fps < 5) {
                                healthBadge.innerHTML = '<span class="status-dot"></span> Degraded';
                                healthBadge.className = 'status-badge warning';
                            } else {
                                healthBadge.innerHTML = '<span class="status-dot"></span> Healthy';
                                healthBadge.className = 'status-badge healthy';
                            }
                        } catch (e) {
                            document.getElementById('inputFpsBadge').textContent = 'IN: -- FPS';
                            document.getElementById('streamFpsBadge').textContent = 'OUT: -- FPS';
                            const healthBadge = document.getElementById('healthBadge');
                            healthBadge.innerHTML = '<span class="status-dot"></span> Offline';
                            healthBadge.className = 'status-badge error';
                        }
                    }

                    setInterval(updateFps, 1000);
                    updateFps();

                    function toggleOverlay() {
                        overlayEnabled = !overlayEnabled;
                        const img = document.getElementById('streamImg');
                        const btn = document.getElementById('toggleBtn');
                        const badge = document.getElementById('statusBadge');
                        const feedInfo = document.getElementById('feedInfo');
                        if (overlayEnabled) {
                            img.src = '/video_feed/' + streamName + '?overlay=true';
                            btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg> Overlay: ON';
                            btn.classList.add('active');
                            badge.innerHTML = '<span class="status-dot"></span> Tracking Enabled';
                            badge.className = 'status-badge overlay';
                            feedInfo.textContent = 'Live Detection Feed';
                        } else {
                            img.src = '/video_feed/' + streamName + '?overlay=false';
                            btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg> Overlay: OFF';
                            btn.classList.remove('active');
                            badge.innerHTML = '<span class="status-dot"></span> Clean Stream';
                            badge.className = 'status-badge clean';
                            feedInfo.textContent = 'Clean Video Feed';
                        }
                    }
                </script>
            </body>
            </html>
            """
            return render_template_string(
                html,
                stream_name=stream_name,
                device_id=device_id,
                location_name=location_name,
            )

        @self.app.route("/video_feed/<stream_name>")
        def video_feed(stream_name):
            """Serve MJPEG video feed for a stream

            Query params:
                overlay: 'true' (default) for annotated stream, 'false' for clean stream
            """
            if not self._is_valid_identifier(stream_name):
                return jsonify({"error": "Invalid stream name"}), 400
            # Check if overlay should be shown (default: true)
            show_overlay = request.args.get("overlay", "true").lower() != "false"

            def generate():
                with self._state_lock:
                    if stream_name not in self.frame_queues:
                        return
                    frame_queue = self.frame_queues[stream_name]

                    # Initialize FPS tracking for this stream
                    if stream_name not in self.stream_stats:
                        self.stream_stats[stream_name] = {
                            "frame_count": 0,
                            "last_reset": time.time(),
                            "stream_fps": 0.0,
                            "input_fps": 0.0,
                        }

                # Paced emission: send one frame every frame_interval to prevent
                # bursty delivery that causes choppy playback in browsers
                frame_interval = 1.0 / VIDEO_FPS
                next_emit = time.time()
                last_frame_data = None

                while True:
                    try:
                        # Sleep until next scheduled emission
                        now = time.time()
                        sleep_time = next_emit - now
                        if sleep_time > 0.001:
                            time.sleep(sleep_time)

                        # Drain queue to get the LATEST frame at this emission tick
                        frame_data = None
                        try:
                            while True:
                                frame_data = frame_queue.get_nowait()
                        except queue.Empty:
                            pass

                        if frame_data is None:
                            if last_frame_data is not None:
                                # No new frame — re-send last frame to maintain cadence
                                frame_data = last_frame_data
                            else:
                                # No frame yet — block briefly for the first one
                                try:
                                    frame_data = frame_queue.get(timeout=1.0)
                                except queue.Empty:
                                    continue

                        last_frame_data = frame_data

                        # Schedule next emission (increment from scheduled time to prevent drift)
                        next_emit += frame_interval
                        if next_emit < time.time() - frame_interval:
                            next_emit = time.time() + frame_interval

                        # Handle both old format (single frame) and new format (dict with clean/annotated)
                        if isinstance(frame_data, dict):
                            frame = (
                                frame_data["annotated"]
                                if show_overlay
                                else frame_data["clean"]
                            )
                            # Update input FPS from processor stats
                            processor_fps = frame_data.get("processor_fps", 0.0)
                            if processor_fps > 0:
                                with self._state_lock:
                                    if stream_name in self.stream_stats:
                                        self.stream_stats[stream_name]["input_fps"] = (
                                            processor_fps
                                        )
                        else:
                            frame = frame_data

                        # Update stream FPS stats (rate of frames delivered to viewer)
                        with self._state_lock:
                            if stream_name in self.stream_stats:
                                stats = self.stream_stats[stream_name]
                                stats["frame_count"] += 1
                                elapsed = time.time() - stats["last_reset"]
                                if elapsed >= 1.0:
                                    stats["stream_fps"] = stats["frame_count"] / elapsed
                                    stats["frame_count"] = 0
                                    stats["last_reset"] = time.time()

                        # Encode frame as JPEG
                        ret, buffer = cv2.imencode(
                            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                        )
                        if not ret:
                            continue

                        jpeg_bytes = buffer.tobytes()

                        # Yield frame in MJPEG format with Content-Length for efficient browser parsing
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: "
                            + str(len(jpeg_bytes)).encode()
                            + b"\r\n\r\n"
                            + jpeg_bytes
                            + b"\r\n"
                        )

                    except queue.Empty:
                        continue
                    except ValueError:
                        # Queue was closed (processor restarted) — stop serving this feed
                        break
                    except Exception as e:
                        self.logger.error(f"Error in video feed for {stream_name}: {e}")
                        break

            return Response(
                generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        @self.app.route("/api/stream/<stream_name>/stats")
        def stream_stats(stream_name):
            """Get stream statistics including FPS, location, resolution, uptime"""
            if not self._is_valid_identifier(stream_name):
                return jsonify({"error": "Invalid stream name"}), 400
            locations = _load_locations()
            with self._state_lock:
                device_info = self.device_info.get(stream_name, {})
                device_id = device_info.get("device_id", stream_name)
                location_name = locations.get(device_id, "")
                is_active = stream_name in self.processors
                start_time = self._processor_start_times.get(stream_name)
                uptime_seconds = round(time.time() - start_time) if start_time else 0

                if stream_name in self.stream_stats:
                    stats = self.stream_stats[stream_name]
                    result = {
                        "stream_name": stream_name,
                        "input_fps": round(stats.get("input_fps", 0), 1),
                        "stream_fps": round(stats.get("stream_fps", 0), 1),
                        "active": is_active,
                        "device_id": device_id,
                        "location_name": location_name,
                        "uptime_seconds": uptime_seconds,
                    }
                else:
                    result = {
                        "stream_name": stream_name,
                        "input_fps": 0,
                        "stream_fps": 0,
                        "active": is_active,
                        "device_id": device_id,
                        "location_name": location_name,
                        "uptime_seconds": uptime_seconds,
                    }

            # Fetch resolution from DB (outside state lock)
            resolution = ""
            try:
                with get_db() as conn:
                    c = conn.cursor()
                    c.execute(
                        "SELECT width, height FROM devices WHERE device_id = ?",
                        (device_id,),
                    )
                    row = c.fetchone()
                    if row:
                        resolution = f"{row['width']}x{row['height']}"
            except Exception:
                pass
            result["resolution"] = resolution
            return jsonify(result)

        @self.app.route("/api/dashboard", methods=["GET"])
        def dashboard_data():
            """Get all registered devices with live processor status for the dashboard."""
            try:
                update_device_status()
                locations = _load_locations()

                with get_db() as conn:
                    c = conn.cursor()
                    c.execute("SELECT * FROM devices ORDER BY device_id")
                    devices_raw = [dict(row) for row in c.fetchall()]

                with self._state_lock:
                    active_streams = set(self.processors.keys())
                    stats_copy = {k: dict(v) for k, v in self.stream_stats.items()}

                # Build set of device_ids owned by mirrors so DB rows don't duplicate them
                mirror_device_ids = set()
                with self._state_lock:
                    for mirror_name in self._mirror_streams:
                        mid = self.device_info.get(mirror_name, {}).get("device_id")
                        if mid:
                            mirror_device_ids.add(mid)

                devices = []
                for dev in devices_raw:
                    device_id = dev["device_id"]
                    stream_name = dev.get("stream_name", "")

                    # Skip DB entries that overlap with mirror streams (injected separately below)
                    if (
                        stream_name in self._mirror_streams
                        or device_id in self._mirror_streams
                        or device_id in mirror_device_ids
                    ):
                        continue

                    is_active = stream_name in active_streams

                    # Only show DB devices if actively running; offline entries come from locations.json
                    if not is_active:
                        continue

                    stats = stats_copy.get(stream_name, {})

                    devices.append(
                        {
                            "device_id": device_id,
                            "stream_name": stream_name,
                            "status": dev.get("status", "offline"),
                            "is_active": is_active,
                            "location_name": locations.get(device_id, ""),
                            "input_fps": round(stats.get("input_fps", 0), 1)
                            if is_active
                            else 0,
                            "stream_fps": round(stats.get("stream_fps", 0), 1)
                            if is_active
                            else 0,
                            "last_seen": dev.get("last_seen", ""),
                            "tailscale_ip": dev.get("tailscale_ip", ""),
                        }
                    )

                # Add all mirror streams (online if processor running, offline if not)
                with self._state_lock:
                    for mirror_name in self._mirror_streams:
                        mirror_device_id = self.device_info.get(mirror_name, {}).get(
                            "device_id", mirror_name
                        )
                        is_active = mirror_name in self.processors
                        mirror_stats = stats_copy.get(mirror_name, {})
                        devices.append(
                            {
                                "device_id": mirror_device_id,
                                "stream_name": mirror_name,
                                "status": "online" if is_active else "offline",
                                "is_active": is_active,
                                "is_mirror": True,
                                "location_name": locations.get(mirror_device_id, ""),
                                "input_fps": round(mirror_stats.get("input_fps", 0), 1)
                                if is_active
                                else 0,
                                "stream_fps": round(
                                    mirror_stats.get("stream_fps", 0), 1
                                )
                                if is_active
                                else 0,
                                "last_seen": "",
                                "tailscale_ip": "",
                            }
                        )

                # Add remaining devices from locations.json that aren't already listed
                # (shows all known devices as offline placeholders on the dev dashboard)
                seen_device_ids = {d["device_id"] for d in devices}
                for loc_device_id, loc_name in locations.items():
                    if loc_device_id not in seen_device_ids:
                        devices.append(
                            {
                                "device_id": loc_device_id,
                                "stream_name": "",
                                "status": "offline",
                                "is_active": False,
                                "location_name": loc_name,
                                "input_fps": 0,
                                "stream_fps": 0,
                                "last_seen": "",
                                "tailscale_ip": "",
                            }
                        )

                # Sort: active first, then offline, alphabetical within each group
                devices.sort(key=lambda d: (not d["is_active"], d["device_id"]))
                online_count = sum(1 for d in devices if d["is_active"])

                return jsonify(
                    {
                        "devices": devices,
                        "online_count": online_count,
                        "total_count": len(devices),
                    }
                )
            except Exception as e:
                self.logger.error(f"Error in /api/dashboard: {e}")
                return jsonify(
                    {
                        "error": "Internal error",
                        "devices": [],
                        "online_count": 0,
                        "total_count": 0,
                    }
                ), 500

        # Web Dashboard Routes
        @self.app.route("/")
        def index():
            """Serve the web dashboard"""
            return send_from_directory(str(STATIC_DIR), "index.html")

        @self.app.route("/<path:filename>")
        def serve_static(filename):
            """Serve static files (CSS, JS, images, etc.)"""
            return send_from_directory(str(STATIC_DIR), filename)

        # Device Management API Routes
        @self.app.route("/api/devices", methods=["GET"])
        def list_devices():
            """List all registered devices"""
            try:
                update_device_status()

                with get_db() as conn:
                    c = conn.cursor()
                    c.execute("SELECT * FROM devices ORDER BY device_id")
                    devices = [dict(row) for row in c.fetchall()]

                return jsonify(devices)

            except Exception as e:
                self.logger.error(f"Error listing devices: {e}")
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route("/api/device/<path:device_id>", methods=["GET"])
        @self._validate_device_id
        def get_device_info(device_id):
            """Get details of a specific device"""
            try:
                with get_db() as conn:
                    c = conn.cursor()
                    c.execute("SELECT * FROM devices WHERE device_id = ?", (device_id,))
                    device = c.fetchone()

                if not device:
                    return jsonify({"error": "Device not found"}), 404

                return jsonify(dict(device))

            except Exception as e:
                self.logger.error(f"Error getting device: {e}")
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route("/api/device/<path:device_id>/restart", methods=["POST"])
        @self.limiter.limit("50000 per minute")
        @self._validate_device_id
        def restart_device(device_id):
            """Restart a device's camera stream service"""
            try:
                with get_db() as conn:
                    c = conn.cursor()
                    c.execute(
                        "SELECT tailscale_ip FROM devices WHERE device_id = ?",
                        (device_id,),
                    )
                    result = c.fetchone()

                if not result or not result["tailscale_ip"]:
                    return jsonify({"error": "Device not found or no IP"}), 404

                ip = result["tailscale_ip"]

                # Get SSH password if available
                password = self.ssh_passwords.get(device_id)

                # Debug logging
                self.logger.info(f"Restart request for device: {device_id}")
                self.logger.info(f"  - Has '@' in device_id: {'@' in device_id}")
                self.logger.info(
                    f"  - Password in ssh_passwords: {device_id in self.ssh_passwords}"
                )
                self.logger.info(f"  - Password value: {'***' if password else 'None'}")

                # Check if password is required but not provided
                if "@" in device_id and not password:
                    self.logger.info("  → Returning 401: Password required")
                    return jsonify(
                        {"error": "Password required", "requires_password": True}
                    ), 401

                # Restart the systemd service
                self.logger.info(f"  → Attempting SSH connection to {ip}")
                success, output = ssh_exec(
                    ip,
                    "sudo systemctl restart camera-stream.service",
                    device_id=device_id,
                    password=password,
                )

                if success:
                    self.logger.info(f"✓ Restarted device {device_id}")
                    return jsonify(
                        {"status": "ok", "message": "Device restarted successfully"}
                    )
                else:
                    self.logger.error(f"✗ Failed to restart {device_id}: {output}")
                    return jsonify({"error": output}), 500

            except Exception as e:
                self.logger.error(f"Error restarting device: {e}")
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route("/api/device/<path:device_id>/stop", methods=["POST"])
        @self.limiter.limit("50000 per minute")
        @self._validate_device_id
        def stop_device(device_id):
            """Stop a device's camera stream service"""
            try:
                with get_db() as conn:
                    c = conn.cursor()
                    c.execute(
                        "SELECT tailscale_ip FROM devices WHERE device_id = ?",
                        (device_id,),
                    )
                    result = c.fetchone()

                if not result or not result["tailscale_ip"]:
                    return jsonify({"error": "Device not found or no IP"}), 404

                ip = result["tailscale_ip"]

                # Get SSH password if available
                password = self.ssh_passwords.get(device_id)

                # Check if password is required but not provided
                if "@" in device_id and not password:
                    return jsonify(
                        {"error": "Password required", "requires_password": True}
                    ), 401

                success, output = ssh_exec(
                    ip,
                    "sudo systemctl stop camera-stream.service",
                    device_id=device_id,
                    password=password,
                )

                if success:
                    self.logger.info(f"Stopped device {device_id}")
                    return jsonify(
                        {"status": "ok", "message": "Device stopped successfully"}
                    )
                else:
                    return jsonify({"error": output}), 500

            except Exception as e:
                self.logger.error(f"Error stopping device: {e}")
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route("/api/device/<path:device_id>/start", methods=["POST"])
        @self.limiter.limit("50000 per minute")
        @self._validate_device_id
        def start_device(device_id):
            """Start a device's camera stream service"""
            try:
                with get_db() as conn:
                    c = conn.cursor()
                    c.execute(
                        "SELECT tailscale_ip FROM devices WHERE device_id = ?",
                        (device_id,),
                    )
                    result = c.fetchone()

                if not result or not result["tailscale_ip"]:
                    return jsonify({"error": "Device not found or no IP"}), 404

                ip = result["tailscale_ip"]

                # Get SSH password if available
                password = self.ssh_passwords.get(device_id)

                # Check if password is required but not provided
                if "@" in device_id and not password:
                    return jsonify(
                        {"error": "Password required", "requires_password": True}
                    ), 401

                success, output = ssh_exec(
                    ip,
                    "sudo systemctl start camera-stream.service",
                    device_id=device_id,
                    password=password,
                )

                if success:
                    self.logger.info(f"Started device {device_id}")
                    return jsonify(
                        {"status": "ok", "message": "Device started successfully"}
                    )
                else:
                    return jsonify({"error": output}), 500

            except Exception as e:
                self.logger.error(f"Error starting device: {e}")
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route("/api/device/<path:device_id>/logs", methods=["GET"])
        @self._validate_device_id
        def get_device_logs(device_id):
            """Get service logs from a device"""
            try:
                with get_db() as conn:
                    c = conn.cursor()
                    c.execute(
                        "SELECT tailscale_ip FROM devices WHERE device_id = ?",
                        (device_id,),
                    )
                    result = c.fetchone()

                if not result or not result["tailscale_ip"]:
                    return jsonify({"error": "Device not found or no IP"}), 404

                ip = result["tailscale_ip"]
                # prevent user dumps
                try:
                    lines = int(request.args.get("lines", "50"))
                    lines = max(1, min(lines, 1000))
                except (ValueError, TypeError):
                    lines = 50

                # Get SSH password if available
                password = self.ssh_passwords.get(device_id)

                # Check if password is required but not provided
                if "@" in device_id and not password:
                    return jsonify(
                        {"error": "Password required", "requires_password": True}
                    ), 401

                success, output = ssh_exec(
                    ip,
                    f"sudo journalctl -u camera-stream.service -n {lines} --no-pager",
                    device_id=device_id,
                    password=password,
                )

                if success:
                    return jsonify({"logs": output})
                else:
                    return jsonify({"error": output}), 500

            except Exception as e:
                self.logger.error(f"Error getting logs: {e}")
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route("/api/device/<path:device_id>/config", methods=["PUT"])
        @self.limiter.limit("50000 per minute")
        @self._validate_device_id
        def update_device_config(device_id):
            """Update device configuration"""
            try:
                data = request.get_json()

                # Reject unknown fields
                ALLOWED_CONFIG_FIELDS = {"width", "height", "fps"}
                if not data or not set(data.keys()).issubset(ALLOWED_CONFIG_FIELDS):
                    return jsonify({"error": "Invalid fields"}), 400

                # Validate input values to prevent shell injection
                if "width" in data:
                    try:
                        width = int(data["width"])
                        if not (160 <= width <= 4096):
                            return jsonify(
                                {"error": "Width must be between 160 and 4096"}
                            ), 400
                        data["width"] = width
                    except (ValueError, TypeError):
                        return jsonify({"error": "Width must be an integer"}), 400

                if "height" in data:
                    try:
                        height = int(data["height"])
                        if not (120 <= height <= 2160):
                            return jsonify(
                                {"error": "Height must be between 120 and 2160"}
                            ), 400
                        data["height"] = height
                    except (ValueError, TypeError):
                        return jsonify({"error": "Height must be an integer"}), 400

                if "fps" in data:
                    try:
                        fps = int(data["fps"])
                        if not (1 <= fps <= 120):
                            return jsonify(
                                {"error": "FPS must be between 1 and 120"}
                            ), 400
                        data["fps"] = fps
                    except (ValueError, TypeError):
                        return jsonify({"error": "FPS must be an integer"}), 400

                with get_db() as conn:
                    c = conn.cursor()
                    c.execute(
                        "SELECT tailscale_ip FROM devices WHERE device_id = ?",
                        (device_id,),
                    )
                    result = c.fetchone()

                    if not result or not result["tailscale_ip"]:
                        return jsonify({"error": "Device not found or no IP"}), 404

                    ip = result["tailscale_ip"]

                    # Get SSH password if available
                    password = self.ssh_passwords.get(device_id)

                    # Check if password is required but not provided
                    if "@" in device_id and not password:
                        return jsonify(
                            {"error": "Password required", "requires_password": True}
                        ), 401

                    # Update database
                    updates = []
                    values = []
                    if "width" in data:
                        updates.append("width = ?")
                        values.append(data["width"])
                    if "height" in data:
                        updates.append("height = ?")
                        values.append(data["height"])
                    if "fps" in data:
                        updates.append("fps = ?")
                        values.append(data["fps"])

                    if updates:
                        values.append(device_id)
                        c.execute(
                            f"UPDATE devices SET {', '.join(updates)} WHERE device_id = ?",
                            values,
                        )
                        conn.commit()

                # Update config file on device
                commands = []
                if "width" in data:
                    commands.append(
                        f"sed -i 's/^WIDTH=.*/WIDTH={data['width']}/' ~/counter-stream/device-info.conf"
                    )
                if "height" in data:
                    commands.append(
                        f"sed -i 's/^HEIGHT=.*/HEIGHT={data['height']}/' ~/counter-stream/device-info.conf"
                    )
                if "fps" in data:
                    commands.append(
                        f"sed -i 's/^FPS=.*/FPS={data['fps']}/' ~/counter-stream/device-info.conf"
                    )

                if commands:
                    command = " && ".join(commands)
                    success, output = ssh_exec(
                        ip, command, device_id=device_id, password=password
                    )

                    if not success:
                        return jsonify(
                            {"error": f"Failed to update config file: {output}"}
                        ), 500

                    # Restart service to apply changes
                    success, output = ssh_exec(
                        ip,
                        "sudo systemctl restart camera-stream.service",
                        device_id=device_id,
                        password=password,
                    )
                    if not success:
                        return jsonify(
                            {"error": f"Config updated but restart failed: {output}"}
                        ), 500

                self.logger.info(f"Updated config for {device_id}")
                return jsonify(
                    {"status": "ok", "message": "Configuration updated successfully"}
                )

            except Exception as e:
                self.logger.error(f"Error updating config: {e}")
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route("/api/device/<path:device_id>/status", methods=["GET"])
        @self._validate_device_id
        def get_device_service_status(device_id):
            """Get current service status from device"""
            try:
                with get_db() as conn:
                    c = conn.cursor()
                    c.execute(
                        "SELECT tailscale_ip FROM devices WHERE device_id = ?",
                        (device_id,),
                    )
                    result = c.fetchone()

                if not result or not result["tailscale_ip"]:
                    return jsonify({"error": "Device not found or no IP"}), 404

                ip = result["tailscale_ip"]

                # Get SSH password if available
                password = self.ssh_passwords.get(device_id)

                # Check if password is required but not provided
                if "@" in device_id and not password:
                    return jsonify(
                        {"error": "Password required", "requires_password": True}
                    ), 401

                success, output = ssh_exec(
                    ip,
                    "sudo systemctl is-active camera-stream.service",
                    device_id=device_id,
                    password=password,
                )

                if success:
                    status = output.strip()
                    return jsonify(
                        {"service_status": status, "is_running": status == "active"}
                    )
                else:
                    return jsonify({"service_status": "unknown", "is_running": False})

            except Exception as e:
                self.logger.error(f"Error getting status: {e}")
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route("/api/server/info", methods=["GET"])
        def server_info():
            """Get server information"""
            try:
                # Get Tailscale IP from environment variable (set in docker-compose)
                # Fall back to running tailscale command if not set
                tailscale_ip = os.environ.get("TAILSCALE_IP")
                if not tailscale_ip:
                    try:
                        result = subprocess.run(
                            ["tailscale", "ip", "-4"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        tailscale_ip = (
                            result.stdout.strip()
                            if result.returncode == 0
                            else "Unknown"
                        )
                    except FileNotFoundError:
                        tailscale_ip = "Unknown"

                return jsonify(
                    {
                        "tailscale_ip": tailscale_ip,
                        "rtsp_port": RTSP_PORT,
                        "orchestrator_port": ORCHESTRATOR_PORT,
                    }
                )
            except Exception as e:
                self.logger.error(f"Error getting server info: {e}")
                return jsonify({"error": "Internal server error"}), 500

        # SSH Password Management Routes
        @self.app.route("/api/device/<path:device_id>/ssh/password", methods=["POST"])
        @self.limiter.limit("50000 per minute")
        @self._validate_device_id
        def set_ssh_password(device_id):
            """Set SSH password for a device (for developer counters)"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "Invalid request body"}), 400
                password = data.get("password", "")

                if not password:
                    return jsonify({"error": "Password is required"}), 400

                # Store password in memory
                self.ssh_passwords[device_id] = password
                self.logger.info(f"SSH password set for device: {device_id}")

                return jsonify(
                    {"status": "ok", "message": "Password stored successfully"}
                )

            except Exception as e:
                self.logger.error(f"Error setting SSH password: {e}")
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route("/api/device/<path:device_id>/ssh/password", methods=["DELETE"])
        @self.limiter.limit("50000 per minute")
        @self._validate_device_id
        def delete_ssh_password(device_id):
            """Delete stored SSH password for a device"""
            try:
                if device_id in self.ssh_passwords:
                    del self.ssh_passwords[device_id]
                    self.logger.info(f"SSH password deleted for device: {device_id}")

                return jsonify({"status": "ok", "message": "Password deleted"})

            except Exception as e:
                self.logger.error(f"Error deleting SSH password: {e}")
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route(
            "/api/device/<path:device_id>/ssh/requires-password", methods=["GET"]
        )
        @self._validate_device_id
        def check_password_required(device_id):
            """Check if device requires SSH password (developer counter)"""
            try:
                # Developer counters have '@' in device_id (e.g., crowdcount1@crowdcount1)
                requires_password = "@" in device_id
                has_password = device_id in self.ssh_passwords

                return jsonify(
                    {
                        "requires_password": requires_password,
                        "has_password": has_password,
                        "device_id": device_id,
                    }
                )

            except Exception as e:
                self.logger.error(f"Error checking password requirement: {e}")
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route("/api/backlog/status", methods=["GET"])
        def backlog_status():
            """Get count backlog status for observability"""
            if self._backlog is None:
                return jsonify({"total_pending": 0, "by_device": {}})
            try:
                return jsonify(
                    {
                        "total_pending": self._backlog.get_count(),
                        "by_device": self._backlog.get_count_by_device(),
                    }
                )
            except Exception as e:
                self.logger.error(f"Error getting backlog status: {e}")
                return jsonify({"error": "Internal server error"}), 500

    def _flush_backlog_loop(self):
        """Background loop that drains the count backlog when the API is reachable."""
        self.logger.info(
            f"Backlog flush thread started (interval={BACKLOG_FLUSH_INTERVAL}s, batch={BACKLOG_FLUSH_BATCH_SIZE})"
        )
        while self.running:
            time.sleep(BACKLOG_FLUSH_INTERVAL)
            if not self.running or self._backlog is None:
                break

            try:
                entries = self._backlog.get_pending(limit=BACKLOG_FLUSH_BATCH_SIZE)
                if not entries:
                    continue

                flushed = 0
                for entry in entries:
                    if not self.running:
                        break
                    self._backlog.mark_attempted(entry["id"])
                    if self._flush_single_entry(entry):
                        self._backlog.mark_completed(entry["id"])
                        flushed += 1
                    else:
                        # API probably down again - stop this cycle, retry next interval
                        break

                if flushed > 0:
                    remaining = self._backlog.get_count()
                    self.logger.info(
                        f"Backlog flush: sent {flushed} entries, {remaining} remaining"
                    )

            except Exception as e:
                self.logger.error(f"Backlog flush error: {e}")

    def _flush_single_entry(self, entry: dict) -> bool:
        """Attempt to send a single backlogged count to the API. Returns True on success."""
        device_id = entry["device_id"]
        password = self.device_info.get(device_id, {}).get("password", "")
        if not password:
            self.logger.warning(
                f"Backlog flush: no password for device {device_id}, skipping"
            )
            return False

        op = entry["operation"]
        if op == "add":
            endpoint = "/add_to_count"
            data = {
                "device_id": device_id,
                "password": password,
                "count_to_add": str(entry["count"]),
            }
        else:
            endpoint = "/sub_from_count"
            data = {
                "device_id": device_id,
                "password": password,
                "count_to_sub": str(entry["count"]),
            }

        # Future timed-API hook: entry['crossing_timestamp'] is available here.
        # When timed endpoints are ready, include the timestamp in the payload.

        try:
            url = f"https://api.crowdcount.tech{endpoint}"
            response = req_lib.post(url, json=data, timeout=10)
            return response.status_code == 200
        except (req_lib.Timeout, req_lib.ConnectionError):
            return False
        except Exception as e:
            self.logger.error(f"Backlog flush unexpected error: {e}")
            return False

    def _detect_available_gpus(self) -> int:
        """Detect the number of available CUDA GPUs.

        Uses NUM_GPUS environment variable as override if set > 1,
        otherwise auto-detects via PyTorch.
        """
        try:
            import torch

            detected = torch.cuda.device_count()
            self.logger.info(f"Detected {detected} CUDA GPU(s)")

            # Use NUM_GPUS as override if explicitly set higher than detected
            # This handles cases where Docker GPU mapping differs from CUDA detection
            if NUM_GPUS > detected:
                self.logger.info(
                    f"NUM_GPUS={NUM_GPUS} overrides detected count ({detected})"
                )
                return NUM_GPUS

            return detected
        except Exception:
            self.logger.warning(
                "Could not detect CUDA GPUs, falling back to NUM_GPUS config"
            )
            return NUM_GPUS

    def start_inference_servers(self):
        """
        Start centralized inference servers, one per GPU.
        Must be called before starting any camera processors.
        """
        from pathlib import Path

        num_gpus = self._detect_available_gpus()
        if num_gpus == 0:
            self.logger.warning(
                "No GPUs detected - processors will use legacy per-process mode"
            )
            return

        # Get model paths from the processor class defaults
        models_dir = Path(__file__).parent.parent.parent / "models"
        model_path = str(models_dir / "crowdcount-v6.pt")
        reid_model_name = "osnet_x0_25"  # Default ReID model

        self.logger.info(f"Starting {num_gpus} inference server(s)...")

        # Pre-allocate result queue pools with per-GPU slot limits
        for gpu_id in range(num_gpus):
            # Get slot limit for this GPU (default to MAX_PROCESSORS if not specified)
            slots_for_gpu = GPU_SLOT_LIMITS.get(gpu_id, MAX_PROCESSORS)

            # Create shared request queue for this GPU
            request_queue = multiprocessing.Queue(maxsize=INFERENCE_QUEUE_MAXSIZE)
            self.inference_request_queues[gpu_id] = request_queue

            # Pre-allocate result queue pool (these are inherited by the child process)
            result_queues = {}
            slot_assignments = {}
            for slot_idx in range(slots_for_gpu):
                slot_name = f"slot_{slot_idx}"
                result_queues[slot_name] = multiprocessing.Queue(maxsize=50)
                slot_assignments[slot_name] = None  # Not assigned yet
            self._result_queue_pools[gpu_id] = result_queues
            self._slot_assignments[gpu_id] = slot_assignments

            # Create heartbeat for liveness monitoring
            heartbeat = multiprocessing.Value("d", 0.0)
            self._inference_heartbeats[gpu_id] = heartbeat

            # Create and start inference server process
            server = InferenceServer(
                gpu_id=gpu_id,
                model_path=model_path,
                reid_model_name=reid_model_name,
                request_queue=request_queue,
                result_queues=result_queues,
                config={
                    "imgsz": INFERENCE_IMGSZ,
                    "batch_size": INFERENCE_BATCH_SIZE,
                    "batch_wait_ms": INFERENCE_BATCH_WAIT_MS,
                    "tensorrt_enabled": TENSORRT_ENABLED,
                    "tensorrt_engine_dir": TENSORRT_ENGINE_DIR,
                    "tensorrt_max_batch": TENSORRT_MAX_BATCH,
                    "tensorrt_warmup_iters": TENSORRT_WARMUP_ITERS,
                },
                heartbeat=heartbeat,
            )

            # Use 'spawn' context to avoid inheriting parent's CUDA context
            # (required for TensorRT — forked CUDA contexts cause illegal memory access)
            ctx = multiprocessing.get_context("spawn")
            proc = ctx.Process(
                target=server.run,
                name=f"InferenceServer-GPU{gpu_id}",
            )
            proc.start()
            self.inference_servers[gpu_id] = proc
            self.logger.info(
                f"Inference server started on GPU {gpu_id} (PID: {proc.pid}, "
                f"{slots_for_gpu} result queue slots pre-allocated)"
            )

    def stop_inference_servers(self):
        """Stop all inference server processes."""
        for gpu_id, proc in list(self.inference_servers.items()):
            self.logger.info(f"Stopping inference server on GPU {gpu_id}...")
            try:
                # Send shutdown request
                if gpu_id in self.inference_request_queues:
                    try:
                        self.inference_request_queues[gpu_id].put_nowait(
                            InferenceRequest(
                                type=RequestType.SHUTDOWN, camera_id="__shutdown__"
                            )
                        )
                    except Exception:
                        pass

                proc.join(timeout=10)
                if proc.is_alive():
                    self.logger.warning(
                        f"Inference server GPU {gpu_id} didn't stop gracefully, killing"
                    )
                    proc.kill()
                    proc.join(timeout=3)
                    if proc.is_alive():
                        self.logger.error(
                            f"Inference server GPU {gpu_id} won't die - abandoning"
                        )
                self.logger.info(f"Inference server GPU {gpu_id} stopped")
            except Exception as e:
                self.logger.error(f"Error stopping inference server GPU {gpu_id}: {e}")

        # Cleanup all queues before clearing
        for gpu_id, req_queue in list(self.inference_request_queues.items()):
            _cleanup_queue(req_queue)

        for gpu_id, result_queues in list(self._result_queue_pools.items()):
            for slot_name, result_queue in result_queues.items():
                _cleanup_queue(result_queue)

        self.inference_servers.clear()
        self.inference_request_queues.clear()
        self._result_queue_pools.clear()
        self._slot_assignments.clear()
        self._inference_heartbeats.clear()

    def _restart_inference_server(self, gpu_id: int):
        """Restart a single inference server on the given GPU.

        Drains the old request queue (and closes it), drains stale results from
        existing result queues (reuses them), creates a new request queue, and
        spawns a fresh InferenceServer process.
        """
        from pathlib import Path

        self.logger.info(f"Restarting inference server on GPU {gpu_id}...")

        # Drain and close the old request queue
        if gpu_id in self.inference_request_queues:
            _cleanup_queue(self.inference_request_queues[gpu_id])

        # Create a new request queue
        request_queue = multiprocessing.Queue(maxsize=INFERENCE_QUEUE_MAXSIZE)
        self.inference_request_queues[gpu_id] = request_queue

        # Drain stale results from existing result queues (reuse them)
        if gpu_id in self._result_queue_pools:
            for slot_name, result_queue in self._result_queue_pools[gpu_id].items():
                try:
                    while True:
                        result_queue.get_nowait()
                except queue.Empty:
                    pass

        # Get model paths
        models_dir = Path(__file__).parent.parent.parent / "models"
        model_path = str(models_dir / "crowdcount-v6.pt")
        reid_model_name = "osnet_x0_25"

        # Create new heartbeat for the restarted server
        heartbeat = multiprocessing.Value("d", 0.0)
        self._inference_heartbeats[gpu_id] = heartbeat

        # Create and start new inference server (reuses existing result queue pool)
        server = InferenceServer(
            gpu_id=gpu_id,
            model_path=model_path,
            reid_model_name=reid_model_name,
            request_queue=request_queue,
            result_queues=self._result_queue_pools[gpu_id],
            config={
                "imgsz": INFERENCE_IMGSZ,
                "batch_size": INFERENCE_BATCH_SIZE,
                "batch_wait_ms": INFERENCE_BATCH_WAIT_MS,
                "tensorrt_enabled": TENSORRT_ENABLED,
                "tensorrt_engine_dir": TENSORRT_ENGINE_DIR,
                "tensorrt_max_batch": TENSORRT_MAX_BATCH,
                "tensorrt_warmup_iters": TENSORRT_WARMUP_ITERS,
            },
            heartbeat=heartbeat,
        )

        # Use 'spawn' context to avoid inheriting parent's CUDA context
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=server.run,
            name=f"InferenceServer-GPU{gpu_id}",
        )
        proc.start()
        self.inference_servers[gpu_id] = proc
        self.logger.info(
            f"Inference server restarted on GPU {gpu_id} (PID: {proc.pid})"
        )

    def _restart_processors_on_gpu(self, gpu_id: int):
        """Restart all camera processors assigned to a GPU after inference server restart.

        After an inference server restart, processors still hold old (closed) queue
        references and will never get results. This stops each affected processor
        and starts a fresh one that gets the new queue references.
        """
        with self._state_lock:
            cameras_on_gpu = [
                cam
                for cam, (gid, slot) in self.camera_gpu_assignments.items()
                if gid == gpu_id
            ]
        self.logger.info(f"Restarting {len(cameras_on_gpu)} processors on GPU {gpu_id}")
        for cam_name in cameras_on_gpu:
            try:
                self.stop_processor(cam_name, reason="inference_server_restart")
                self.start_processor(cam_name)
            except Exception as e:
                self.logger.error(
                    f"Error restarting processor {cam_name} after GPU {gpu_id} recovery: {e}"
                )

    def _assign_camera_to_gpu(self, stream_name: str) -> tuple:
        """Assign a camera to a GPU using priority-based allocation.

        GPUs are filled in priority order (GPU_PRIORITY_ORDER config).
        Default: GPU 1 first (25 slots), then overflow to GPU 0 (10 slots).

        Returns:
            (gpu_id, slot_name) or (-1, None) if no servers/slots available
        """
        with self._state_lock:
            if not self.inference_servers:
                return -1, None

            # Check if this camera already has an assignment (race condition protection)
            if stream_name in self.camera_gpu_assignments:
                existing = self.camera_gpu_assignments[stream_name]
                self.logger.warning(
                    f"Camera {stream_name} already assigned to GPU {existing[0]}, slot {existing[1]} - reusing"
                )
                return existing

            # Try GPUs in priority order (fill primary GPU first, then overflow)
            available_gpus = [
                g for g in GPU_PRIORITY_ORDER if g in self.inference_servers
            ]
            # Also include any GPUs not in priority list (fallback)
            for gpu_id in self.inference_servers.keys():
                if gpu_id not in available_gpus:
                    available_gpus.append(gpu_id)

            for idx, gpu_id in enumerate(available_gpus):
                # Count current usage on this GPU
                used_slots = sum(
                    1 for c in self._slot_assignments[gpu_id].values() if c is not None
                )
                total_slots = len(self._slot_assignments[gpu_id])

                # Find a free slot on this GPU
                slot_name = None
                for sname, camera_id in self._slot_assignments[gpu_id].items():
                    if camera_id is None:
                        slot_name = sname
                        break

                if slot_name is not None:
                    # Found a free slot - use it
                    self._slot_assignments[gpu_id][slot_name] = stream_name
                    self.camera_gpu_assignments[stream_name] = (gpu_id, slot_name)
                    used_slots += 1  # Include the one we just assigned
                    if idx > 0:
                        # We had to overflow to a secondary GPU
                        self.logger.info(
                            f"Assigned camera {stream_name} to OVERFLOW GPU {gpu_id}, slot {slot_name} ({used_slots}/{total_slots} slots used)"
                        )
                    else:
                        self.logger.info(
                            f"Assigned camera {stream_name} to GPU {gpu_id}, slot {slot_name} ({used_slots}/{total_slots} slots used)"
                        )
                    return gpu_id, slot_name
                else:
                    # This GPU is full, log and try next
                    self.logger.debug(
                        f"GPU {gpu_id} full ({used_slots}/{total_slots} slots) - trying next GPU"
                    )

            # No free slots on any GPU
            self.logger.error(
                f"No free result queue slots on any GPU for camera {stream_name}"
            )
            return -1, None

    def _has_inference_servers(self) -> bool:
        """Check if centralized inference servers are running."""
        with self._state_lock:
            if not self.inference_servers:
                return False
            servers_copy = list(self.inference_servers.values())
        return all(proc.is_alive() for proc in servers_copy)

    def start_processor(self, stream_name: str):
        """
        Start a processor for a stream

        Args:
            stream_name: Name of the stream to process
        """
        # Check if already processing or being started
        with self._state_lock:
            if stream_name in self.processors:
                proc = self.processors[stream_name]
                if proc.is_alive():
                    self.logger.info(f"Processor for {stream_name} is already running")
                    return
                else:
                    self.logger.warning(
                        f"Processor for {stream_name} was dead, restarting"
                    )
                    # Release lock before calling stop_processor (which acquires it)
                    needs_stop = True
            else:
                needs_stop = False

            # Check if another thread is already starting a processor for this stream
            if not needs_stop and stream_name in self._starting_processors:
                self.logger.info(
                    f"Processor for {stream_name} is being started by another thread - skipping"
                )
                return

            # Check processor limit
            if len(self.processors) >= MAX_PROCESSORS:
                self.logger.warning(
                    f"Maximum processors ({MAX_PROCESSORS}) reached, cannot start processor for {stream_name}"
                )
                return

            # Mark as starting to prevent duplicate attempts
            self._starting_processors[stream_name] = time.time()

        if needs_stop:
            self.stop_processor(stream_name, reason="start_processor_replacing_dead")

        # Create RTSP URL
        rtsp_url = f"rtsp://{RTSP_HOST}:{RTSP_PORT}/{stream_name}"

        # Get device info if available
        with self._state_lock:
            device_info = self.device_info.get(stream_name, {})

        self.logger.info(f"Starting processor for stream: {stream_name}")

        is_mirror = stream_name in self._mirror_streams

        try:
            # Create API client for status updates (skip for mirrors)
            device_id = device_info.get("device_id", stream_name)
            password = device_info.get("password", "")
            api_client = CrowdCountAPIClient(
                device_id=device_id,
                password=password,
                base_url="https://api.crowdcount.tech",
                timeout=5,
            )

            # Create frame queue for live viewing
            frame_queue = multiprocessing.Queue(maxsize=15)  # 1 second buffer added

            with self._state_lock:
                if not is_mirror:
                    self.api_clients[stream_name] = api_client
                self.frame_queues[stream_name] = frame_queue

            # Update device status to online (skip for mirrors - no API calls)
            if not is_mirror:
                self._submit_status_update(api_client, "online", stream_name)

            # Set up centralized inference queues if servers are running
            inference_request_queue = None
            inference_result_queue = None

            if self._has_inference_servers():
                self.logger.info(f"Assigning {stream_name} to GPU...")
                gpu_id, slot_name = self._assign_camera_to_gpu(stream_name)
                self.logger.info(
                    f"Assignment result for {stream_name}: gpu_id={gpu_id}, slot_name={slot_name}"
                )
                if gpu_id >= 0 and slot_name is not None:
                    inference_request_queue = self.inference_request_queues[gpu_id]
                    inference_result_queue = self._result_queue_pools[gpu_id][slot_name]
                    # Drain stale results from slot before reuse
                    drained = 0
                    try:
                        while True:
                            inference_result_queue.get_nowait()
                            drained += 1
                    except queue.Empty:
                        pass
                    if drained:
                        self.logger.info(
                            f"Drained {drained} stale results from {slot_name} for {stream_name}"
                        )
                    self.logger.info(
                        f"Sending REGISTER for {stream_name} to GPU {gpu_id}, slot {slot_name}"
                    )
                    # Send REGISTER request so inference server maps camera_id -> slot
                    try:
                        inference_request_queue.put(
                            InferenceRequest(
                                type=RequestType.REGISTER,
                                camera_id=stream_name,
                                slot_name=slot_name,
                            ),
                            timeout=5.0,
                        )
                    except queue.Full:
                        self.logger.error(
                            f"Inference queue full - cannot register {stream_name}. "
                            f"Inference server may be dead. FALLING BACK TO LEGACY MODE: "
                            f"processor will load its own YOLO+ReID models on GPU (high memory usage)"
                        )
                        # Release the GPU slot we just assigned
                        with self._state_lock:
                            self.camera_gpu_assignments.pop(stream_name, None)
                            if (
                                gpu_id in self._slot_assignments
                                and slot_name in self._slot_assignments[gpu_id]
                            ):
                                self._slot_assignments[gpu_id][slot_name] = None
                        # Fall back to legacy mode (no centralized inference)
                        inference_request_queue = None
                        inference_result_queue = None
                    else:
                        self.logger.info(
                            f"Camera {stream_name} using centralized inference on GPU {gpu_id} (slot {slot_name})"
                        )
                else:
                    self.logger.warning(
                        f"Failed to assign {stream_name} to GPU - FALLING BACK TO LEGACY MODE: "
                        f"processor will load its own YOLO+ReID models on GPU (high memory usage)"
                    )

            # Create processor instance with device info, frame queue, and inference queues
            # Use try/except to handle processor classes that don't accept all kwargs
            processor_heartbeat = multiprocessing.Value("d", time.time())
            try:
                processor_kwargs = {
                    "device_info": device_info,
                    "frame_queue": frame_queue,
                    "processor_heartbeat": processor_heartbeat,
                }
                if inference_request_queue is not None:
                    processor_kwargs["inference_request_queue"] = (
                        inference_request_queue
                    )
                    processor_kwargs["inference_result_queue"] = inference_result_queue
                    processor_kwargs["inference_slot_name"] = slot_name
                processor = self.processor_class(
                    stream_name, rtsp_url, **processor_kwargs
                )
            except TypeError:
                # Fallback for processor classes that don't accept all kwargs
                processor = self.processor_class(
                    stream_name, rtsp_url, device_info=device_info
                )

            # Start in separate process for isolation
            proc = multiprocessing.Process(
                target=processor.run, name=f"Processor-{stream_name}"
            )
            proc.start()

            with self._state_lock:
                self.processors[stream_name] = proc
                self._processor_heartbeats[stream_name] = processor_heartbeat
                self._processor_start_times[stream_name] = time.time()
                self._starting_processors.pop(stream_name, None)
            self.logger.info(f"Processor started for {stream_name} (PID: {proc.pid})")

        except Exception as e:
            self.logger.error(f"Error starting processor for {stream_name}: {e}")
            # Clean up GPU assignment if registered but failed to start
            with self._state_lock:
                self._starting_processors.pop(stream_name, None)
                assignment = self.camera_gpu_assignments.pop(stream_name, None)
                if assignment:
                    gpu_id, slot_name = assignment
                    if (
                        gpu_id in self._slot_assignments
                        and slot_name in self._slot_assignments[gpu_id]
                    ):
                        self._slot_assignments[gpu_id][slot_name] = None
                    # Send UNREGISTER
                    if gpu_id in self.inference_request_queues:
                        try:
                            self.inference_request_queues[gpu_id].put(
                                InferenceRequest(
                                    type=RequestType.UNREGISTER,
                                    camera_id=stream_name,
                                ),
                                timeout=2.0,
                            )
                        except Exception as ue:
                            self.logger.warning(
                                f"Could not send UNREGISTER for {stream_name}: {ue}"
                            )
                # Pop queue for cleanup outside lock
                frame_queue = self.frame_queues.pop(stream_name, None)
                self.api_clients.pop(stream_name, None)

            # Clean up frame queue OUTSIDE lock (join_thread can block)
            if frame_queue:
                _cleanup_queue(frame_queue)

    def stop_processor(self, stream_name: str, reason: str = "unknown"):
        """
        Stop a processor for a stream

        Args:
            stream_name: Name of the stream
            reason: Why the processor is being stopped (for diagnostic logging)
        """
        with self._state_lock:
            if stream_name not in self.processors:
                self.logger.warning(f"No processor found for {stream_name}")
                return
            proc = self.processors[stream_name]

        self.logger.info(f"Stopping processor for {stream_name} (reason: {reason})")

        try:
            # Terminate process
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)

                # Force kill if still alive
                if proc.is_alive():
                    self.logger.warning(
                        f"Processor for {stream_name} didn't stop gracefully, killing"
                    )
                    proc.kill()
                    proc.join(timeout=3)  # Timeout to prevent blocking forever
                    if proc.is_alive():
                        self.logger.error(
                            f"Processor for {stream_name} won't die even after kill - abandoning"
                        )

            self.logger.info(f"Processor stopped for {stream_name}")

        except Exception as e:
            self.logger.error(f"Error stopping processor for {stream_name}: {e}")

        finally:
            # Update device status to offline
            with self._state_lock:
                api_client = self.api_clients.pop(stream_name, None)
            if api_client:
                self._submit_status_update(api_client, "offline", stream_name)

            with self._state_lock:
                # Clean up processor entry (use pop to avoid KeyError)
                self.processors.pop(stream_name, None)
                self._processor_heartbeats.pop(stream_name, None)
                self._processor_start_times.pop(stream_name, None)
                # Pop frame queue for cleanup outside lock (join_thread can block)
                frame_queue = self.frame_queues.pop(stream_name, None)
                # Clean up stream stats
                self.stream_stats.pop(stream_name, None)
                # Clean up inference assignment and send UNREGISTER
                assignment = self.camera_gpu_assignments.pop(stream_name, None)
                if assignment is not None:
                    gpu_id, slot_name = assignment
                    # Release the slot
                    if (
                        gpu_id in self._slot_assignments
                        and slot_name in self._slot_assignments[gpu_id]
                    ):
                        self._slot_assignments[gpu_id][slot_name] = None
                    # Tell inference server to unregister this camera
                    if gpu_id in self.inference_request_queues:
                        try:
                            self.inference_request_queues[gpu_id].put(
                                InferenceRequest(
                                    type=RequestType.UNREGISTER,
                                    camera_id=stream_name,
                                ),
                                timeout=2.0,
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Could not send UNREGISTER for {stream_name}: {e}"
                            )

            # Clean up frame queue OUTSIDE lock (join_thread can block)
            if frame_queue:
                _cleanup_queue(frame_queue)
            # Keep device info for reconnection (don't delete credentials)

    def _submit_status_update(self, api_client, status: str, stream_name: str = ""):
        """Submit a status update with a done-callback that logs failures."""

        def _on_done(future, _stream=stream_name, _status=status):
            try:
                future.result()
            except Exception as e:
                self.logger.error(
                    f"Status update to '{_status}' FAILED for {_stream}: {e}"
                )

        try:
            fut = self._status_update_executor.submit(api_client.update_status, status)
            fut.add_done_callback(_on_done)
        except RuntimeError:
            # Executor already shut down
            self.logger.warning(
                f"Cannot submit status update to '{status}' for {stream_name} - executor shut down"
            )

    def _handle_stream_interruption(self, stream_name: str):
        """
        Handle stream interruption with reconnect grace period.
        Does NOT kill the processor — lets it exit naturally when the jitter buffer
        declares staleness (~15s) and consecutive failures accumulate (~2s more).
        This avoids the SIGTERM-during-FFmpeg race condition.
        """
        with self._state_lock:
            if stream_name in self.pending_reconnect:
                # Already pending, just update timestamp
                self.pending_reconnect[stream_name]["stop_time"] = time.time()
                return

            # Preserve API client for later offline update
            api_client = self.api_clients.pop(stream_name, None)

            # Guard against orphan entries: if no api_client and no processor, already cleaned up
            if api_client is None and stream_name not in self.processors:
                self.logger.debug(
                    f"Stream {stream_name} interruption ignored - already cleaned up"
                )
                return

            # Add to pending reconnect — processor will die on its own
            self.pending_reconnect[stream_name] = {
                "stop_time": time.time(),
                "api_client": api_client,
            }

        self.logger.info(
            f"Stream {stream_name} added to reconnect queue (timeout: {STREAM_RECONNECT_TIMEOUT}s) - processor will exit naturally"
        )

    def _handle_dead_processor(
        self, stream_name: str, detected_proc, active_streams: list
    ):
        """
        Handle a single dead processor: cleanup and conditional restart.
        Called from monitor_streams via ThreadPoolExecutor for parallel processing.

        Args:
            stream_name: Name of the stream whose processor died
            detected_proc: The dead process object detected by the monitor
            active_streams: Pre-fetched list of active MediaMTX streams
        """
        try:
            # Re-check: skip if a DIFFERENT processor was started since our snapshot
            with self._state_lock:
                current_proc = self.processors.get(stream_name)
                if current_proc is not None and current_proc is not detected_proc:
                    return  # A different (new) processor replaced the stuck one
                if stream_name in self._starting_processors:
                    return
                in_pending = stream_name in self.pending_reconnect
            if in_pending:
                self._cleanup_dead_processor(stream_name, detected_proc=detected_proc)
                return

            self.stop_processor(stream_name, reason="monitor_dead_processor")

            # Check if stream is still active, restart processor (with limit)
            if stream_name in active_streams:
                # Track restart failures to prevent infinite loops
                with self._state_lock:
                    if stream_name not in self._restart_failures:
                        self._restart_failures[stream_name] = {
                            "count": 0,
                            "last_failure": 0,
                        }

                    failures = self._restart_failures[stream_name]
                    max_restarts = 3

                    # Reset failure count if last failure was more than 10 minutes ago
                    if time.time() - failures["last_failure"] > 300:
                        failures["count"] = 0

                    should_restart = failures["count"] < max_restarts
                    if should_restart:
                        failures["count"] += 1
                        failures["last_failure"] = time.time()
                        restart_attempt = failures["count"]

                    # Get API client for offline notification if needed
                    api_client = (
                        self.api_clients.get(stream_name)
                        if not should_restart
                        else None
                    )

                if should_restart:
                    self.logger.info(
                        f"Stream {stream_name} is still active, restarting processor "
                        f"(attempt {restart_attempt}/{max_restarts})"
                    )
                    self.start_processor(stream_name)
                else:
                    self.logger.warning(
                        f"Stream {stream_name} has failed {max_restarts} consecutive restarts - "
                        f"giving up until stream reconnects. Device may be sending no video data."
                    )
                    # Mark device offline since we can't process its stream
                    if api_client:
                        self._submit_status_update(api_client, "offline", stream_name)
        except Exception as e:
            self.logger.error(
                f"Error handling dead processor for {stream_name}: {e}", exc_info=True
            )

    def _cleanup_dead_processor(self, stream_name: str, detected_proc=None):
        """
        Clean up a dead processor's resources without marking the device offline.
        Used by monitor_streams when a dead processor is in pending_reconnect.
        Releases GPU slot, cleans frame queue, sends UNREGISTER.

        Args:
            detected_proc: The specific process object detected as dead/stuck.
                          If provided, cleanup is skipped only if a DIFFERENT process
                          has replaced it (allowing stuck but alive processes to be cleaned up).
        """
        with self._state_lock:
            # Guard: if a new processor was started since we decided to clean up, bail out
            current_proc = self.processors.get(stream_name)
            if (
                current_proc is not None
                and (detected_proc is None or current_proc is not detected_proc)
                and current_proc.is_alive()
            ):
                self.logger.info(
                    f"Skipping dead processor cleanup for {stream_name} - new processor already running"
                )
                return
            if stream_name in self._starting_processors:
                self.logger.info(
                    f"Skipping dead processor cleanup for {stream_name} - new processor is being started"
                )
                return
            # Clean up processor entry
            proc = self.processors.pop(stream_name, None)
            self._processor_heartbeats.pop(stream_name, None)
            self._processor_start_times.pop(stream_name, None)
            # Pop frame queue for cleanup outside lock (join_thread can block)
            frame_queue = self.frame_queues.pop(stream_name, None)
            # Clean up stream stats
            self.stream_stats.pop(stream_name, None)
            # Clean up inference assignment and send UNREGISTER
            assignment = self.camera_gpu_assignments.pop(stream_name, None)
            if assignment is not None:
                gpu_id, slot_name = assignment
                if (
                    gpu_id in self._slot_assignments
                    and slot_name in self._slot_assignments[gpu_id]
                ):
                    self._slot_assignments[gpu_id][slot_name] = None
                if gpu_id in self.inference_request_queues:
                    try:
                        self.inference_request_queues[gpu_id].put(
                            InferenceRequest(
                                type=RequestType.UNREGISTER,
                                camera_id=stream_name,
                            ),
                            timeout=2.0,
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Could not send UNREGISTER for {stream_name}: {e}"
                        )

        # Terminate stuck processor if still alive (outside lock to avoid blocking)
        if proc is not None and proc.is_alive():
            self.logger.info(f"Terminating stuck processor for {stream_name}")
            try:
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    self.logger.warning(
                        f"Stuck processor for {stream_name} did not terminate, killing"
                    )
                    proc.kill()
                    proc.join(timeout=2)
            except Exception as e:
                self.logger.warning(
                    f"Error terminating stuck processor for {stream_name}: {e}"
                )

        # Clean up frame queue OUTSIDE lock (join_thread can block)
        if frame_queue:
            _cleanup_queue(frame_queue)
        self.logger.info(
            f"Cleaned up dead processor for {stream_name} (pending reconnect)"
        )

    def _check_pending_reconnects(self):
        """Check pending streams - mark offline if timeout expired, restart if reconnected."""
        current_time = time.time()
        active_streams = self.mediamtx.get_active_streams()
        streams_to_remove = []
        streams_to_restart = []
        streams_to_mark_offline = []

        with self._state_lock:
            pending_copy = list(self.pending_reconnect.items())

        for stream_name, info in pending_copy:
            elapsed = current_time - info["stop_time"]

            # Stream came back - restart processor
            if stream_name in active_streams:
                self.logger.info(
                    f"Stream {stream_name} reappeared after {elapsed:.1f}s - restarting processor"
                )
                streams_to_restart.append(stream_name)
                streams_to_remove.append(stream_name)

            # Timeout expired - mark offline
            elif elapsed > STREAM_RECONNECT_TIMEOUT:
                self.logger.warning(
                    f"Stream {stream_name} reconnect timeout ({STREAM_RECONNECT_TIMEOUT}s) - marking offline"
                )
                # Now mark offline
                if info.get("api_client"):
                    streams_to_mark_offline.append(info["api_client"])
                streams_to_remove.append(stream_name)

        # Mark offline outside the lock
        for api_client in streams_to_mark_offline:
            self._submit_status_update(api_client, "offline")

        with self._state_lock:
            for stream_name in streams_to_remove:
                self.pending_reconnect.pop(stream_name, None)

        # Restart processors outside the lock (start_processor acquires its own lock)
        # Don't restore old API client - start_processor creates fresh one with current credentials
        for stream_name in streams_to_restart:
            self.start_processor(stream_name)

    def monitor_streams(self):
        """
        Monitor stream health and restart dead processors
        Runs in a background thread
        """
        self.logger.info("Stream monitor started")
        _monitor_iteration = 0

        while self.running:
            try:
                _monitor_iteration += 1
                with self._monitor_heartbeat_lock:
                    self._monitor_heartbeat = time.time()

                # Periodic resource monitoring (every 10 iterations = ~150s with 15s interval)
                if _monitor_iteration % 10 == 0:
                    fd_count = _get_fd_count()
                    with self._state_lock:
                        proc_count = len(self.processors)
                    self.logger.info(
                        f"Resource stats: FDs={fd_count}, threads={threading.active_count()}, processors={proc_count}"
                    )
                    if fd_count > 50000:
                        self.logger.warning(f"HIGH FD COUNT: {fd_count}")

                # Check MediaMTX health
                mediamtx_healthy = self.mediamtx.is_healthy()
                if not mediamtx_healthy:
                    self.logger.error("MediaMTX is not healthy!")

                # Check if MediaMTX needs restart due to consecutive API failures
                if self.mediamtx.needs_restart():
                    self.logger.warning(
                        "MediaMTX API unresponsive - attempting restart"
                    )
                    if self.mediamtx.restart():
                        self.logger.info("MediaMTX restarted successfully")
                    else:
                        self.logger.error("Failed to restart MediaMTX")
                    # Skip the rest of this iteration to let MediaMTX stabilize
                    time.sleep(STREAM_HEALTH_CHECK_INTERVAL)
                    continue

                # Check inference server health (process death + heartbeat staleness)
                with self._state_lock:
                    servers_copy = list(self.inference_servers.items())
                for gpu_id, server_proc in servers_copy:
                    needs_restart = False
                    if not server_proc.is_alive():
                        self.logger.error(
                            f"Inference server on GPU {gpu_id} died! Restarting..."
                        )
                        needs_restart = True
                    elif gpu_id in self._inference_heartbeats:
                        hb = self._inference_heartbeats[gpu_id]
                        if hb.value > 0 and time.time() - hb.value > 30:
                            self.logger.error(
                                f"Inference server on GPU {gpu_id} stuck! "
                                f"No heartbeat for {time.time() - hb.value:.1f}s - force-killing..."
                            )
                            try:
                                server_proc.kill()
                                server_proc.join(timeout=5)
                            except Exception:
                                pass
                            needs_restart = True

                    if needs_restart:
                        try:
                            self._restart_inference_server(gpu_id)
                            # Restart all processors on this GPU — they hold stale queue refs
                            self._restart_processors_on_gpu(gpu_id)
                        except Exception as e:
                            self.logger.error(
                                f"Failed to restart inference server on GPU {gpu_id}: {e}"
                            )

                # Check processor health
                dead_processors = []  # List of (stream_name, proc) tuples
                with self._state_lock:
                    processors_copy = list(self.processors.items())
                for stream_name, proc in processors_copy:
                    if not proc.is_alive():
                        self.logger.warning(
                            f"Processor for {stream_name} died unexpectedly"
                        )
                        dead_processors.append((stream_name, proc))
                    elif stream_name in self._processor_heartbeats:
                        hb = self._processor_heartbeats[stream_name]
                        if hb.value > 0 and time.time() - hb.value > 60:
                            self.logger.warning(
                                f"Processor for {stream_name} stuck — no heartbeat for "
                                f"{time.time() - hb.value:.1f}s, restarting"
                            )
                            dead_processors.append((stream_name, proc))

                # Clean up dead processors in parallel
                if dead_processors:
                    # Fetch active streams ONCE (not per-stream)
                    active_streams = self.mediamtx.get_active_streams()

                    self.logger.info(
                        f"Processing {len(dead_processors)} dead processors in parallel"
                    )
                    max_workers = min(len(dead_processors), 8)
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {}
                        for stream_name, detected_proc in dead_processors:
                            future = executor.submit(
                                self._handle_dead_processor,
                                stream_name,
                                detected_proc,
                                active_streams,
                            )
                            futures[future] = stream_name

                        # Wait with periodic heartbeat updates to prevent watchdog timeout
                        deadline = time.time() + 60
                        while time.time() < deadline:
                            with self._monitor_heartbeat_lock:
                                self._monitor_heartbeat = time.time()
                            done, not_done = futures_wait(futures, timeout=5)
                            if not not_done:
                                break
                        else:
                            for f in not_done:
                                self.logger.warning(
                                    f"Dead processor handling timed out for {futures[f]}"
                                )

                # Clean up stale _starting_processors entries (stuck for > 30s)
                with self._state_lock:
                    now = time.time()
                    stale_starting = [
                        name
                        for name, start_time in self._starting_processors.items()
                        if now - start_time > 30
                    ]
                    for name in stale_starting:
                        self._starting_processors.pop(name, None)
                        self.logger.warning(
                            f"Cleared stale _starting_processors entry for {name} (stuck > 30s)"
                        )

                # Get active streams and ensure processors exist
                # Note: If MediaMTX API is having issues, get_active_streams() returns empty list
                # We should NOT treat this as "all streams stopped" - only trust webhook notifications
                active_streams = self.mediamtx.get_active_streams()

                # Only process new streams if API call was successful (consecutive failures = 0)
                if self.mediamtx.get_consecutive_failures() == 0:
                    for stream_name in active_streams:
                        with self._state_lock:
                            already_processing = stream_name in self.processors
                            pending = stream_name in self.pending_reconnect
                            # Check if this stream has hit its restart limit
                            failures = self._restart_failures.get(stream_name, {})
                            in_cooldown = (
                                failures.get("count", 0) >= 3
                                and time.time() - failures.get("last_failure", 0) < 300
                            )
                            cooldown_expired = (
                                failures.get("count", 0) >= 3 and not in_cooldown
                            )
                            if cooldown_expired:
                                # Cooldown expired - reset and try again
                                self._restart_failures.pop(stream_name, None)

                        if not already_processing and not pending:
                            if in_cooldown:
                                # Still in cooldown - don't restart
                                continue
                            self.logger.info(
                                f"Found active stream without processor: {stream_name}"
                            )
                            self.start_processor(stream_name)
                else:
                    # API having issues - log but don't make assumptions about stream state
                    self.logger.debug(
                        f"MediaMTX API has {self.mediamtx.get_consecutive_failures()} consecutive failures - "
                        "skipping stream discovery this cycle"
                    )

                # Check streams pending reconnection
                # Always check pending reconnects - timeout logic uses timestamps, not MediaMTX state
                # Even if MediaMTX API is having issues, we still need to mark devices offline after timeout
                # to prevent them from staying in limbo forever
                api_failures = self.mediamtx.get_consecutive_failures()
                if api_failures > 0:
                    time_since_last_success = (
                        time.time() - self.mediamtx.get_last_successful_api_call()
                    )
                    self.logger.debug(
                        f"MediaMTX API has {api_failures} failures, {time_since_last_success:.1f}s since last success - "
                        "still checking pending reconnects for timeout processing"
                    )
                self._check_pending_reconnects()

                # Proactive stale session sweep (every 3rd iteration ~45s)
                # Kick MediaMTX sessions for paths that are "ready" but have no healthy processor
                if (
                    _monitor_iteration % 3 == 0
                    and self.mediamtx.get_consecutive_failures() == 0
                ):
                    try:
                        all_paths = self.mediamtx.get_paths()
                        with self._state_lock:
                            healthy_streams = {
                                name
                                for name, proc in self.processors.items()
                                if proc.is_alive()
                            }
                            pending_streams = set(self.pending_reconnect.keys())
                            cooldown_streams = {
                                name
                                for name, info in self._restart_failures.items()
                                if info.get("count", 0) >= 3
                                and time.time() - info.get("last_failure", 0) < 300
                            }

                            starting_streams = set(self._starting_processors.keys())

                        for path_info in all_paths:
                            path_name = path_info.get("name", "")
                            if not path_name:
                                continue
                            if not path_info.get("ready", False):
                                continue
                            # Skip if processor is currently being started
                            if path_name in starting_streams:
                                self.logger.info(
                                    f"Stale sweep: skipping {path_name} (processor starting)"
                                )
                                continue
                            # Skip if we have a healthy processor, pending reconnect, or in restart cooldown
                            if (
                                path_name in healthy_streams
                                or path_name in pending_streams
                                or path_name in cooldown_streams
                            ):
                                continue
                            self.logger.warning(
                                f"Stale session detected: path {path_name} is ready in MediaMTX "
                                f"but has no healthy processor - kicking sessions"
                            )
                            self.mediamtx.kick_all_sessions_for_path(path_name)
                    except Exception as e:
                        self.logger.error(f"Error in stale session sweep: {e}")

                # Periodic cleanup of stale restart failure entries (older than 10 minutes)
                current_time = time.time()
                with self._state_lock:
                    stale = [
                        n
                        for n, i in self._restart_failures.items()
                        if current_time - i.get("last_failure", 0) > 600
                    ]
                    for name in stale:
                        self._restart_failures.pop(name, None)

            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")

            # Sleep before next check
            time.sleep(STREAM_HEALTH_CHECK_INTERVAL)

        self.logger.info("Stream monitor stopped")

    def _monitor_watchdog(self):
        """Watchdog thread that restarts the monitor thread if it hangs."""
        while self.running:
            time.sleep(30)
            with self._monitor_heartbeat_lock:
                last_beat = self._monitor_heartbeat
            if last_beat == 0.0:
                continue  # Monitor hasn't started yet
            stale_seconds = time.time() - last_beat
            if stale_seconds > 90:
                self.logger.critical(
                    f"Monitor thread appears hung! No heartbeat for {stale_seconds:.0f}s. Restarting monitor thread."
                )
                # The old thread is stuck; start a fresh one
                self.monitor_thread = threading.Thread(
                    target=self.monitor_streams, daemon=True
                )
                self.monitor_thread.start()
                # Reset heartbeat so we don't immediately fire again
                with self._monitor_heartbeat_lock:
                    self._monitor_heartbeat = time.time()

    def start(self):
        """Start the orchestrator"""
        self.logger.info("=" * 60)
        self.logger.info("CrowdCount Counter Server - Stream Orchestrator")
        self.logger.info("=" * 60)

        # Start MediaMTX
        if not self.mediamtx.start():
            self.logger.error("Failed to start MediaMTX, cannot continue")
            return False

        self.running = True

        # Start centralized inference servers (one per GPU)
        self.start_inference_servers()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_streams, daemon=True)
        self.monitor_thread.start()

        # Start monitor watchdog thread
        self._watchdog_thread = threading.Thread(
            target=self._monitor_watchdog, daemon=True
        )
        self._watchdog_thread.start()

        # Initialize count backlog and start flush thread
        try:
            self._backlog = CountBacklog(str(BACKLOG_DB_PATH))
            pending = self._backlog.get_count()
            if pending > 0:
                self.logger.warning(
                    f"Count backlog has {pending} entries from previous session - will flush when API is reachable"
                )
            else:
                self.logger.info("Count backlog initialized (empty)")
            self._backlog_flush_thread = threading.Thread(
                target=self._flush_backlog_loop, daemon=True
            )
            self._backlog_flush_thread.start()
        except Exception as e:
            self.logger.error(f"Failed to initialize count backlog: {e}")
            self._backlog = None

        # Suppress Flask/Werkzeug logging
        import logging as flask_logging

        flask_log = flask_logging.getLogger("werkzeug")
        flask_log.setLevel(flask_logging.ERROR)

        # Start Flask app in a separate thread
        flask_thread = threading.Thread(
            target=lambda: self.app.run(
                host=ORCHESTRATOR_HOST,
                port=ORCHESTRATOR_PORT,
                debug=False,
                use_reloader=False,
            ),
            daemon=True,
        )
        flask_thread.start()

        self.logger.info(
            f"Orchestrator API listening on {ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}"
        )
        self.logger.info(f"Using processor class: {self.processor_class.__name__}")
        self.logger.info(f"Max concurrent processors: {MAX_PROCESSORS}")
        if self.inference_servers:
            self.logger.info(
                f"Inference servers: {len(self.inference_servers)} GPU(s) [centralized mode]"
            )
        else:
            self.logger.info("Inference servers: none [legacy per-process GPU mode]")
        self.logger.info("=" * 60)
        self.logger.info("Ready to receive streams!")
        self.logger.info("Raspberry Pis can now connect via RTSP")
        self.logger.info("=" * 60)

        return True

    def stop(self):
        """Stop the orchestrator"""
        self.logger.info("Stopping orchestrator...")
        self.running = False

        # Log remaining backlog count before shutdown
        if self._backlog is not None:
            try:
                remaining = self._backlog.get_count()
                if remaining > 0:
                    self.logger.warning(
                        f"Shutting down with {remaining} backlogged count entries - will be flushed on next startup"
                    )
                self._backlog.close()
            except Exception as e:
                self.logger.error(f"Error closing backlog: {e}")

        # Set all devices offline (using executor for consistency)
        for stream_name, api_client in list(self.api_clients.items()):
            self._submit_status_update(api_client, "offline", stream_name)

        # Also mark pending_reconnect devices offline
        for stream_name, info in list(self.pending_reconnect.items()):
            api_client = info.get("api_client")
            if api_client:
                self._submit_status_update(api_client, "offline", stream_name)

        # Stop all processors BEFORE shutting down executor
        # (stop_processor's finally block may submit to executor)
        for stream_name in list(self.processors.keys()):
            self.stop_processor(stream_name, reason="orchestrator_shutdown")

        # NOW shutdown executor after all processors are stopped
        self._status_update_executor.shutdown(wait=True, cancel_futures=False)

        # Stop inference servers
        self.stop_inference_servers()

        # Stop MediaMTX
        self.mediamtx.stop()

        # Wait for monitor thread
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self.logger.info("Orchestrator stopped")

    def run(self):
        """Run the orchestrator (blocking)"""
        if not self.start():
            return

        # Setup signal handlers
        def signal_handler(sig, frame):
            if self._stopping:
                return
            self._stopping = True
            self.logger.info("\nReceived shutdown signal")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Keep running
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Received interrupt")
        finally:
            self.stop()


def main():
    """Main entry point - for testing only, use run_server.py instead"""
    # Note: When using this directly, set multiprocessing start method before importing
    # Create orchestrator with example processor
    # Later, you can replace ExampleProcessor with your actual processor class
    orchestrator = StreamOrchestrator(processor_class=ExampleProcessor)

    # Run
    orchestrator.run()


if __name__ == "__main__":
    main()
