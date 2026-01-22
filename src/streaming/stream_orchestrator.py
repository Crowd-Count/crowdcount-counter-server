"""
Stream Orchestrator
Main orchestrator that manages MediaMTX and stream processors
Automatically spawns a processor for each new stream
"""

import os
import sys
import time
import logging
import threading
import signal
from pathlib import Path
from typing import Dict, Type, Optional
from flask import Flask, request, jsonify, Response, render_template_string, send_from_directory
from flask_cors import CORS
import multiprocessing
import cv2
import queue
import sqlite3
import paramiko
import subprocess
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.crowdcount_api_client import CrowdCountAPIClient
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
    VIDEO_HEIGHT
)
from streaming.mediamtx_manager import MediaMTXManager
from processors import BaseStreamProcessor, ExampleProcessor


# Database helper functions
def init_db():
    """Create database tables if they don't exist"""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute(f'''
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
    ''')
    conn.commit()
    conn.close()


def get_db():
    """Get database connection"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def ssh_exec(ip: str, command: str, device_id: str = None, password: str = None) -> tuple:
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
            if '@' in device_id:
                # Developer counter: extract username from device_id
                username = device_id.split('@')[0]
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
                look_for_keys=False
            )
        else:
            # Try key-based authentication (for normal counters or if no password)
            ssh.connect(
                ip,
                username=username,
                key_filename=SSH_KEY_PATH,
                timeout=10,
                allow_agent=True,
                look_for_keys=True
            )

        stdin, stdout, stderr = ssh.exec_command(command)
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
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
    conn = get_db()
    c = conn.cursor()

    # Mark devices as offline if not seen in 60 seconds
    threshold = datetime.now() - timedelta(seconds=60)
    c.execute('''
        UPDATE devices
        SET status = 'offline'
        WHERE last_seen < ? AND status != 'offline'
    ''', (threshold,))

    conn.commit()
    conn.close()


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

        # Track FPS stats per stream for viewer display
        # Format: {stream_name: {'frame_count': int, 'last_reset': float, 'fps': float}}
        self.stream_stats: Dict[str, dict] = {}

        self.running = False
        self.monitor_thread = None

        # Setup logging
        self.logger = logging.getLogger("Orchestrator")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, LOG_LEVEL))

        # Initialize database
        init_db()

        # Flask app for receiving MediaMTX webhooks and device management
        self.app = Flask(__name__, static_folder=str(STATIC_DIR))
        CORS(self.app)
        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes for MediaMTX webhooks"""

        @self.app.route('/stream/ready', methods=['POST'])
        def stream_ready():
            """Called by MediaMTX when a stream becomes ready"""
            data = request.get_json()
            stream_name = data.get('stream', '')

            if stream_name:
                # Cancel any pending reconnect timeout (stream came back)
                if stream_name in self.pending_reconnect:
                    elapsed = time.time() - self.pending_reconnect[stream_name]['stop_time']
                    self.logger.info(f"Stream {stream_name} reconnected after {elapsed:.1f}s - canceling offline timeout")
                    del self.pending_reconnect[stream_name]

                # Check if this stream is already being processed (avoid duplicate calls)
                if stream_name in self.processors and self.processors[stream_name].is_alive():
                    # Stream already has a processor running, ignore
                    return jsonify({'status': 'ok', 'message': 'Already processing'}), 200

                # Get client IP (Tailscale IP)
                client_ip = request.remote_addr

                # Get device info from pre-registration, or use defaults
                if stream_name in self.device_info:
                    device_id = self.device_info[stream_name].get('device_id', stream_name)
                    password = self.device_info[stream_name].get('password', '')
                else:
                    # No pre-registration, use stream name as device_id
                    device_id = stream_name
                    password = ''
                    self.device_info[stream_name] = {
                        'device_id': device_id,
                        'password': password,
                        'timestamp': time.time()
                    }

                # Update device status in database
                try:
                    conn = get_db()
                    c = conn.cursor()
                    c.execute('SELECT * FROM devices WHERE device_id = ?', (device_id,))
                    existing = c.fetchone()

                    if existing:
                        # Update existing device
                        c.execute('''
                            UPDATE devices
                            SET stream_name = ?, tailscale_ip = ?, last_seen = ?, status = 'online'
                            WHERE device_id = ?
                        ''', (stream_name, client_ip, datetime.now(), device_id))
                    else:
                        # Insert new device
                        c.execute('''
                            INSERT INTO devices (device_id, stream_name, tailscale_ip, last_seen, status)
                            VALUES (?, ?, ?, ?, 'online')
                        ''', (device_id, stream_name, client_ip, datetime.now()))

                    conn.commit()
                    conn.close()
                except Exception as e:
                    self.logger.error(f"Error updating device status in database: {e}")

                # Detailed logging for testing
                self.logger.info("=" * 60)
                self.logger.info(f"🔗 NEW DEVICE CONNECTED")
                self.logger.info(f"   Stream Name: {stream_name}")
                self.logger.info(f"   Device ID:   {device_id}")
                self.logger.info(f"   IP Address:  {client_ip}")
                self.logger.info(f"   Password:    {password if password else '(none)'}")
                self.logger.info("=" * 60)

                self.start_processor(stream_name)
                return jsonify({'status': 'ok'}), 200

            return jsonify({'status': 'error', 'message': 'No stream name provided'}), 400

        @self.app.route('/stream/stopped', methods=['POST'])
        def stream_stopped():
            """Called by MediaMTX when a stream stops - uses grace period before marking offline"""
            data = request.get_json()
            stream_name = data.get('stream', '')

            if stream_name:
                self.logger.info(f"Received stream stopped notification: {stream_name}")

                # Stop the processor but don't mark offline yet - use grace period
                self._handle_stream_interruption(stream_name)

                return jsonify({'status': 'ok'}), 200

            return jsonify({'status': 'error', 'message': 'No stream name provided'}), 400

        @self.app.route('/auth', methods=['POST'])
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
            user = data.get('user', '')
            password = data.get('password', '')
            action = data.get('action', '')

            if not user or not password:
                self.logger.warning(f"✗ Auth attempt with missing credentials from {request.remote_addr}")
                return '', 401  # Unauthorized

            # Validate against CrowdCount API
            is_valid = CrowdCountAPIClient.check_password(user, password)

            if is_valid:
                self.logger.info(f"✓ Device authenticated: {user} (action: {action})")
                return '', 200  # OK
            else:
                self.logger.warning(f"✗ Authentication failed for {user}: Invalid credentials")
                return '', 401  # Unauthorized

        @self.app.route('/status', methods=['GET'])
        def status():
            """Get orchestrator status"""
            return jsonify({
                'running': self.running,
                'mediamtx_healthy': self.mediamtx.is_healthy(),
                'active_streams': list(self.processors.keys()),
                'processor_count': len(self.processors),
                'max_processors': MAX_PROCESSORS,
                'device_info': {
                    stream: {
                        'device_id': info['device_id'],
                        'has_password': bool(info.get('password'))
                    }
                    for stream, info in self.device_info.items()
                }
            }), 200

        @self.app.route('/processors', methods=['GET'])
        def processors():
            """Get processor information"""
            processor_info = {}
            for stream_name, proc in self.processors.items():
                processor_info[stream_name] = {
                    'alive': proc.is_alive(),
                    'pid': proc.pid
                }
            return jsonify(processor_info), 200

        @self.app.route('/device/register', methods=['POST'])
        def register_device():
            """
            Register device information for a stream.
            Validates device credentials before allowing registration.
            """
            data = request.get_json()
            stream_name = data.get('stream_name', '')
            device_id = data.get('device_id', stream_name)
            password = data.get('password', '')

            if not stream_name:
                return jsonify({'status': 'error', 'message': 'stream_name is required'}), 400

            if not device_id or not password:
                self.logger.warning(f"✗ Registration attempt with missing credentials from {request.remote_addr}")
                return jsonify({'status': 'error', 'message': 'device_id and password are required'}), 400

            # Validate credentials against CrowdCount API
            self.logger.info(f"Validating credentials for {device_id}...")
            is_valid = CrowdCountAPIClient.check_password(device_id, password)
            if not is_valid:
                self.logger.warning(f"✗ Registration failed for {device_id}: Invalid credentials")
                return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401
            self.logger.info(f"✓ Credentials valid for {device_id}")

            # Get client IP (Tailscale IP)
            client_ip = request.remote_addr

            # Store device info in memory
            self.device_info[stream_name] = {
                'device_id': device_id,
                'password': password,
                'timestamp': time.time()
            }

            # Store device info in database
            try:
                conn = get_db()
                c = conn.cursor()

                # Check if device exists
                c.execute('SELECT * FROM devices WHERE device_id = ?', (device_id,))
                existing = c.fetchone()

                if existing:
                    # Update existing device
                    c.execute('''
                        UPDATE devices
                        SET stream_name = ?, tailscale_ip = ?, last_seen = ?, status = 'online'
                        WHERE device_id = ?
                    ''', (stream_name, client_ip, datetime.now(), device_id))
                else:
                    # Insert new device
                    c.execute('''
                        INSERT INTO devices (device_id, stream_name, tailscale_ip, last_seen, status)
                        VALUES (?, ?, ?, ?, 'online')
                    ''', (device_id, stream_name, client_ip, datetime.now()))

                conn.commit()
                conn.close()

                self.logger.info(f"Registered device: {device_id} (stream: {stream_name}, IP: {client_ip})")
                return jsonify({'status': 'ok', 'stream_name': stream_name, 'device_id': device_id}), 200

            except Exception as e:
                self.logger.error(f"Error registering device in database: {e}")
                # Still return success for backward compatibility
                return jsonify({'status': 'ok', 'stream_name': stream_name, 'device_id': device_id}), 200

        @self.app.route('/view/<stream_name>')
        def view_stream(stream_name):
            """Serve HTML viewer for a stream"""
            html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{{ stream_name }} - CrowdCount PROD</title>
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
                        background: var(--success);
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
                <div class="env-banner">PRODUCTION</div>
                <div class="app">
                    <header class="header">
                        <h1>Stream Viewer</h1>
                        <p class="stream-id">{{ stream_name }}</p>
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
                        <span id="fpsBadge" class="status-badge fps">-- FPS</span>
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

                    // Update FPS display
                    async function updateFps() {
                        try {
                            const response = await fetch('/api/stream/' + streamName + '/stats');
                            const data = await response.json();
                            const fpsText = data.fps.toFixed(1) + ' FPS';
                            document.getElementById('fpsBadge').textContent = fpsText;
                        } catch (e) {
                            document.getElementById('fpsBadge').textContent = '-- FPS';
                        }
                    }

                    // Start FPS polling
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
            return render_template_string(html, stream_name=stream_name)

        @self.app.route('/video_feed/<stream_name>')
        def video_feed(stream_name):
            """Serve MJPEG video feed for a stream

            Query params:
                overlay: 'true' (default) for annotated stream, 'false' for clean stream
            """
            # Check if overlay should be shown (default: true)
            show_overlay = request.args.get('overlay', 'true').lower() != 'false'

            def generate():
                if stream_name not in self.frame_queues:
                    return

                frame_queue = self.frame_queues[stream_name]

                # Initialize FPS tracking for this stream
                if stream_name not in self.stream_stats:
                    self.stream_stats[stream_name] = {
                        'frame_count': 0,
                        'last_reset': time.time(),
                        'fps': 0.0
                    }

                while True:
                    try:
                        # Get frame data from queue with timeout
                        frame_data = frame_queue.get(timeout=1.0)

                        # Drain queue to get the LATEST frame (reduces lag)
                        while True:
                            try:
                                frame_data = frame_queue.get_nowait()
                            except queue.Empty:
                                break

                        # Handle both old format (single frame) and new format (dict with clean/annotated)
                        if isinstance(frame_data, dict):
                            frame = frame_data['annotated'] if show_overlay else frame_data['clean']
                        else:
                            # Backwards compatibility: single frame
                            frame = frame_data

                        # Update FPS stats
                        stats = self.stream_stats[stream_name]
                        stats['frame_count'] += 1
                        elapsed = time.time() - stats['last_reset']
                        if elapsed >= 1.0:
                            stats['fps'] = stats['frame_count'] / elapsed
                            stats['frame_count'] = 0
                            stats['last_reset'] = time.time()

                        # Encode frame as JPEG
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if not ret:
                            continue

                        # Yield frame in MJPEG format
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                    except queue.Empty:
                        # No frame available, continue waiting
                        continue
                    except Exception as e:
                        self.logger.error(f"Error in video feed for {stream_name}: {e}")
                        break

            return Response(generate(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/api/stream/<stream_name>/stats')
        def stream_stats(stream_name):
            """Get stream statistics including FPS"""
            if stream_name in self.stream_stats:
                stats = self.stream_stats[stream_name]
                return jsonify({
                    'stream_name': stream_name,
                    'fps': round(stats['fps'], 1),
                    'active': stream_name in self.processors
                })
            return jsonify({
                'stream_name': stream_name,
                'fps': 0,
                'active': stream_name in self.processors
            })

        # Web Dashboard Routes
        @self.app.route('/')
        def index():
            """Serve the web dashboard"""
            return send_from_directory(str(STATIC_DIR), 'index.html')

        @self.app.route('/<path:filename>')
        def serve_static(filename):
            """Serve static files (CSS, JS, images, etc.)"""
            return send_from_directory(str(STATIC_DIR), filename)

        # Device Management API Routes
        @self.app.route('/api/devices', methods=['GET'])
        def list_devices():
            """List all registered devices"""
            try:
                update_device_status()

                conn = get_db()
                c = conn.cursor()
                c.execute('SELECT * FROM devices ORDER BY device_id')
                devices = [dict(row) for row in c.fetchall()]
                conn.close()

                return jsonify(devices)

            except Exception as e:
                self.logger.error(f"Error listing devices: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/device/<path:device_id>', methods=['GET'])
        def get_device_info(device_id):
            """Get details of a specific device"""
            try:
                conn = get_db()
                c = conn.cursor()
                c.execute('SELECT * FROM devices WHERE device_id = ?', (device_id,))
                device = c.fetchone()
                conn.close()

                if not device:
                    return jsonify({'error': 'Device not found'}), 404

                return jsonify(dict(device))

            except Exception as e:
                self.logger.error(f"Error getting device: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/device/<path:device_id>/restart', methods=['POST'])
        def restart_device(device_id):
            """Restart a device's camera stream service"""
            try:
                conn = get_db()
                c = conn.cursor()
                c.execute('SELECT tailscale_ip FROM devices WHERE device_id = ?', (device_id,))
                result = c.fetchone()
                conn.close()

                if not result or not result['tailscale_ip']:
                    return jsonify({'error': 'Device not found or no IP'}), 404

                ip = result['tailscale_ip']

                # Get SSH password if available
                password = self.ssh_passwords.get(device_id)

                # Debug logging
                self.logger.info(f"Restart request for device: {device_id}")
                self.logger.info(f"  - Has '@' in device_id: {'@' in device_id}")
                self.logger.info(f"  - Password in ssh_passwords: {device_id in self.ssh_passwords}")
                self.logger.info(f"  - Password value: {'***' if password else 'None'}")

                # Check if password is required but not provided
                if '@' in device_id and not password:
                    self.logger.info(f"  → Returning 401: Password required")
                    return jsonify({
                        'error': 'Password required',
                        'requires_password': True
                    }), 401

                # Restart the systemd service
                self.logger.info(f"  → Attempting SSH connection to {ip}")
                success, output = ssh_exec(ip, 'sudo systemctl restart camera-stream.service',
                                          device_id=device_id, password=password)

                if success:
                    self.logger.info(f"✓ Restarted device {device_id}")
                    return jsonify({'status': 'ok', 'message': 'Device restarted successfully'})
                else:
                    self.logger.error(f"✗ Failed to restart {device_id}: {output}")
                    return jsonify({'error': output}), 500

            except Exception as e:
                self.logger.error(f"Error restarting device: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/device/<path:device_id>/stop', methods=['POST'])
        def stop_device(device_id):
            """Stop a device's camera stream service"""
            try:
                conn = get_db()
                c = conn.cursor()
                c.execute('SELECT tailscale_ip FROM devices WHERE device_id = ?', (device_id,))
                result = c.fetchone()
                conn.close()

                if not result or not result['tailscale_ip']:
                    return jsonify({'error': 'Device not found or no IP'}), 404

                ip = result['tailscale_ip']

                # Get SSH password if available
                password = self.ssh_passwords.get(device_id)

                # Check if password is required but not provided
                if '@' in device_id and not password:
                    return jsonify({
                        'error': 'Password required',
                        'requires_password': True
                    }), 401

                success, output = ssh_exec(ip, 'sudo systemctl stop camera-stream.service',
                                          device_id=device_id, password=password)

                if success:
                    self.logger.info(f"Stopped device {device_id}")
                    return jsonify({'status': 'ok', 'message': 'Device stopped successfully'})
                else:
                    return jsonify({'error': output}), 500

            except Exception as e:
                self.logger.error(f"Error stopping device: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/device/<path:device_id>/start', methods=['POST'])
        def start_device(device_id):
            """Start a device's camera stream service"""
            try:
                conn = get_db()
                c = conn.cursor()
                c.execute('SELECT tailscale_ip FROM devices WHERE device_id = ?', (device_id,))
                result = c.fetchone()
                conn.close()

                if not result or not result['tailscale_ip']:
                    return jsonify({'error': 'Device not found or no IP'}), 404

                ip = result['tailscale_ip']

                # Get SSH password if available
                password = self.ssh_passwords.get(device_id)

                # Check if password is required but not provided
                if '@' in device_id and not password:
                    return jsonify({
                        'error': 'Password required',
                        'requires_password': True
                    }), 401

                success, output = ssh_exec(ip, 'sudo systemctl start camera-stream.service',
                                          device_id=device_id, password=password)

                if success:
                    self.logger.info(f"Started device {device_id}")
                    return jsonify({'status': 'ok', 'message': 'Device started successfully'})
                else:
                    return jsonify({'error': output}), 500

            except Exception as e:
                self.logger.error(f"Error starting device: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/device/<path:device_id>/logs', methods=['GET'])
        def get_device_logs(device_id):
            """Get service logs from a device"""
            try:
                conn = get_db()
                c = conn.cursor()
                c.execute('SELECT tailscale_ip FROM devices WHERE device_id = ?', (device_id,))
                result = c.fetchone()
                conn.close()

                if not result or not result['tailscale_ip']:
                    return jsonify({'error': 'Device not found or no IP'}), 404

                ip = result['tailscale_ip']
                lines = request.args.get('lines', '50')

                # Get SSH password if available
                password = self.ssh_passwords.get(device_id)

                # Check if password is required but not provided
                if '@' in device_id and not password:
                    return jsonify({
                        'error': 'Password required',
                        'requires_password': True
                    }), 401

                success, output = ssh_exec(ip, f'sudo journalctl -u camera-stream.service -n {lines} --no-pager',
                                          device_id=device_id, password=password)

                if success:
                    return jsonify({'logs': output})
                else:
                    return jsonify({'error': output}), 500

            except Exception as e:
                self.logger.error(f"Error getting logs: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/device/<path:device_id>/config', methods=['PUT'])
        def update_device_config(device_id):
            """Update device configuration"""
            try:
                data = request.get_json()

                # Validate input values to prevent shell injection
                if 'width' in data:
                    try:
                        width = int(data['width'])
                        if not (160 <= width <= 4096):
                            return jsonify({'error': 'Width must be between 160 and 4096'}), 400
                        data['width'] = width
                    except (ValueError, TypeError):
                        return jsonify({'error': 'Width must be an integer'}), 400

                if 'height' in data:
                    try:
                        height = int(data['height'])
                        if not (120 <= height <= 2160):
                            return jsonify({'error': 'Height must be between 120 and 2160'}), 400
                        data['height'] = height
                    except (ValueError, TypeError):
                        return jsonify({'error': 'Height must be an integer'}), 400

                if 'fps' in data:
                    try:
                        fps = int(data['fps'])
                        if not (1 <= fps <= 120):
                            return jsonify({'error': 'FPS must be between 1 and 120'}), 400
                        data['fps'] = fps
                    except (ValueError, TypeError):
                        return jsonify({'error': 'FPS must be an integer'}), 400

                conn = get_db()
                c = conn.cursor()
                c.execute('SELECT tailscale_ip FROM devices WHERE device_id = ?', (device_id,))
                result = c.fetchone()

                if not result or not result['tailscale_ip']:
                    conn.close()
                    return jsonify({'error': 'Device not found or no IP'}), 404

                ip = result['tailscale_ip']

                # Get SSH password if available
                password = self.ssh_passwords.get(device_id)

                # Check if password is required but not provided
                if '@' in device_id and not password:
                    conn.close()
                    return jsonify({
                        'error': 'Password required',
                        'requires_password': True
                    }), 401

                # Update database
                updates = []
                values = []
                if 'width' in data:
                    updates.append('width = ?')
                    values.append(data['width'])
                if 'height' in data:
                    updates.append('height = ?')
                    values.append(data['height'])
                if 'fps' in data:
                    updates.append('fps = ?')
                    values.append(data['fps'])

                if updates:
                    values.append(device_id)
                    c.execute(f'UPDATE devices SET {", ".join(updates)} WHERE device_id = ?', values)
                    conn.commit()

                conn.close()

                # Update config file on device
                commands = []
                if 'width' in data:
                    commands.append(f"sed -i 's/^WIDTH=.*/WIDTH={data['width']}/' ~/counter-stream/device-info.conf")
                if 'height' in data:
                    commands.append(f"sed -i 's/^HEIGHT=.*/HEIGHT={data['height']}/' ~/counter-stream/device-info.conf")
                if 'fps' in data:
                    commands.append(f"sed -i 's/^FPS=.*/FPS={data['fps']}/' ~/counter-stream/device-info.conf")

                if commands:
                    command = ' && '.join(commands)
                    success, output = ssh_exec(ip, command, device_id=device_id, password=password)

                    if not success:
                        return jsonify({'error': f'Failed to update config file: {output}'}), 500

                    # Restart service to apply changes
                    success, output = ssh_exec(ip, 'sudo systemctl restart camera-stream.service',
                                              device_id=device_id, password=password)
                    if not success:
                        return jsonify({'error': f'Config updated but restart failed: {output}'}), 500

                self.logger.info(f"Updated config for {device_id}")
                return jsonify({'status': 'ok', 'message': 'Configuration updated successfully'})

            except Exception as e:
                self.logger.error(f"Error updating config: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/device/<path:device_id>/status', methods=['GET'])
        def get_device_service_status(device_id):
            """Get current service status from device"""
            try:
                conn = get_db()
                c = conn.cursor()
                c.execute('SELECT tailscale_ip FROM devices WHERE device_id = ?', (device_id,))
                result = c.fetchone()
                conn.close()

                if not result or not result['tailscale_ip']:
                    return jsonify({'error': 'Device not found or no IP'}), 404

                ip = result['tailscale_ip']

                # Get SSH password if available
                password = self.ssh_passwords.get(device_id)

                # Check if password is required but not provided
                if '@' in device_id and not password:
                    return jsonify({
                        'error': 'Password required',
                        'requires_password': True
                    }), 401

                success, output = ssh_exec(ip, 'sudo systemctl is-active camera-stream.service',
                                          device_id=device_id, password=password)

                if success:
                    status = output.strip()
                    return jsonify({'service_status': status, 'is_running': status == 'active'})
                else:
                    return jsonify({'service_status': 'unknown', 'is_running': False})

            except Exception as e:
                self.logger.error(f"Error getting status: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/server/info', methods=['GET'])
        def server_info():
            """Get server information"""
            try:
                # Get Tailscale IP from environment variable (set in docker-compose)
                # Fall back to running tailscale command if not set
                tailscale_ip = os.environ.get('TAILSCALE_IP')
                if not tailscale_ip:
                    try:
                        result = subprocess.run(['tailscale', 'ip', '-4'], capture_output=True, text=True, timeout=5)
                        tailscale_ip = result.stdout.strip() if result.returncode == 0 else 'Unknown'
                    except FileNotFoundError:
                        tailscale_ip = 'Unknown'

                return jsonify({
                    'tailscale_ip': tailscale_ip,
                    'rtsp_port': RTSP_PORT,
                    'orchestrator_port': ORCHESTRATOR_PORT
                })
            except Exception as e:
                self.logger.error(f"Error getting server info: {e}")
                return jsonify({'error': str(e)}), 500

        # SSH Password Management Routes
        @self.app.route('/api/device/<path:device_id>/ssh/password', methods=['POST'])
        def set_ssh_password(device_id):
            """Set SSH password for a device (for developer counters)"""
            try:
                data = request.get_json()
                password = data.get('password', '')

                if not password:
                    return jsonify({'error': 'Password is required'}), 400

                # Store password in memory
                self.ssh_passwords[device_id] = password
                self.logger.info(f"SSH password set for device: {device_id}")

                return jsonify({'status': 'ok', 'message': 'Password stored successfully'})

            except Exception as e:
                self.logger.error(f"Error setting SSH password: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/device/<path:device_id>/ssh/password', methods=['DELETE'])
        def delete_ssh_password(device_id):
            """Delete stored SSH password for a device"""
            try:
                if device_id in self.ssh_passwords:
                    del self.ssh_passwords[device_id]
                    self.logger.info(f"SSH password deleted for device: {device_id}")

                return jsonify({'status': 'ok', 'message': 'Password deleted'})

            except Exception as e:
                self.logger.error(f"Error deleting SSH password: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/device/<path:device_id>/ssh/requires-password', methods=['GET'])
        def check_password_required(device_id):
            """Check if device requires SSH password (developer counter)"""
            try:
                # Developer counters have '@' in device_id (e.g., crowdcount1@crowdcount1)
                requires_password = '@' in device_id
                has_password = device_id in self.ssh_passwords

                return jsonify({
                    'requires_password': requires_password,
                    'has_password': has_password,
                    'device_id': device_id
                })

            except Exception as e:
                self.logger.error(f"Error checking password requirement: {e}")
                return jsonify({'error': str(e)}), 500

    def start_processor(self, stream_name: str):
        """
        Start a processor for a stream

        Args:
            stream_name: Name of the stream to process
        """
        # Check if already processing
        if stream_name in self.processors:
            proc = self.processors[stream_name]
            if proc.is_alive():
                self.logger.info(f"Processor for {stream_name} is already running")
                return
            else:
                self.logger.warning(f"Processor for {stream_name} was dead, restarting")
                self.stop_processor(stream_name)

        # Check processor limit
        if len(self.processors) >= MAX_PROCESSORS:
            self.logger.warning(f"Maximum processors ({MAX_PROCESSORS}) reached, cannot start processor for {stream_name}")
            return

        # Create RTSP URL
        rtsp_url = f"rtsp://{RTSP_HOST}:{RTSP_PORT}/{stream_name}"

        # Get device info if available
        device_info = self.device_info.get(stream_name, {})

        self.logger.info(f"Starting processor for stream: {stream_name}")

        try:
            # Create API client for status updates
            device_id = device_info.get('device_id', stream_name)
            password = device_info.get('password', '')
            api_client = CrowdCountAPIClient(
                device_id=device_id,
                password=password,
                base_url='https://api.crowdcount.tech',
                timeout=5
            )
            self.api_clients[stream_name] = api_client

            # Update device status to online
            threading.Thread(target=api_client.update_status, args=('online',), daemon=True).start()

            # Create frame queue for live viewing
            frame_queue = multiprocessing.Queue(maxsize=2)  # Small queue to avoid memory issues
            self.frame_queues[stream_name] = frame_queue

            # Create processor instance with device info and frame queue
            processor = self.processor_class(stream_name, rtsp_url, device_info=device_info, frame_queue=frame_queue)

            # Start in separate process for isolation
            proc = multiprocessing.Process(target=processor.run, name=f"Processor-{stream_name}")
            proc.start()

            self.processors[stream_name] = proc
            self.logger.info(f"Processor started for {stream_name} (PID: {proc.pid})")

        except Exception as e:
            self.logger.error(f"Error starting processor for {stream_name}: {e}")

    def stop_processor(self, stream_name: str):
        """
        Stop a processor for a stream

        Args:
            stream_name: Name of the stream
        """
        if stream_name not in self.processors:
            self.logger.warning(f"No processor found for {stream_name}")
            return

        self.logger.info(f"Stopping processor for {stream_name}")

        proc = self.processors[stream_name]

        try:
            # Terminate process
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)

                # Force kill if still alive
                if proc.is_alive():
                    self.logger.warning(f"Processor for {stream_name} didn't stop gracefully, killing")
                    proc.kill()
                    proc.join()

            self.logger.info(f"Processor stopped for {stream_name}")

        except Exception as e:
            self.logger.error(f"Error stopping processor for {stream_name}: {e}")

        finally:
            # Update device status to offline
            if stream_name in self.api_clients:
                api_client = self.api_clients[stream_name]
                threading.Thread(target=api_client.update_status, args=('offline',), daemon=True).start()
                del self.api_clients[stream_name]

            # Clean up processor entry (use pop to avoid KeyError)
            self.processors.pop(stream_name, None)
            # Clean up frame queue
            if stream_name in self.frame_queues:
                del self.frame_queues[stream_name]
            # Keep device info for reconnection (don't delete credentials)

    def _handle_stream_interruption(self, stream_name: str):
        """
        Handle stream interruption with reconnect grace period.
        Stops the processor but doesn't mark offline immediately - waits for possible reconnection.
        """
        if stream_name in self.pending_reconnect:
            # Already pending, just update timestamp
            self.pending_reconnect[stream_name]['stop_time'] = time.time()
            return

        # Stop processor but preserve API client for later offline update
        api_client = self.api_clients.pop(stream_name, None)

        if stream_name in self.processors:
            proc = self.processors[stream_name]
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                    if proc.is_alive():
                        proc.kill()
                        proc.join()
                self.logger.info(f"Processor stopped for {stream_name}")
            except Exception as e:
                self.logger.error(f"Error stopping processor for {stream_name}: {e}")
            finally:
                self.processors.pop(stream_name, None)

        # Clean up frame queue
        if stream_name in self.frame_queues:
            del self.frame_queues[stream_name]

        # Add to pending reconnect instead of marking offline immediately
        self.pending_reconnect[stream_name] = {
            'stop_time': time.time(),
            'api_client': api_client
        }
        self.logger.info(f"Stream {stream_name} added to reconnect queue (timeout: {STREAM_RECONNECT_TIMEOUT}s)")

    def _check_pending_reconnects(self):
        """Check pending streams - mark offline if timeout expired, restart if reconnected."""
        current_time = time.time()
        active_streams = self.mediamtx.get_active_streams()
        streams_to_remove = []

        for stream_name, info in self.pending_reconnect.items():
            elapsed = current_time - info['stop_time']

            # Stream came back - restart processor
            if stream_name in active_streams:
                self.logger.info(f"Stream {stream_name} reappeared after {elapsed:.1f}s - restarting processor")
                # Restore API client before starting processor
                if info.get('api_client'):
                    self.api_clients[stream_name] = info['api_client']
                self.start_processor(stream_name)
                streams_to_remove.append(stream_name)

            # Timeout expired - mark offline
            elif elapsed > STREAM_RECONNECT_TIMEOUT:
                self.logger.warning(f"Stream {stream_name} reconnect timeout ({STREAM_RECONNECT_TIMEOUT}s) - marking offline")
                # Now mark offline
                if info.get('api_client'):
                    threading.Thread(
                        target=info['api_client'].update_status,
                        args=('offline',),
                        daemon=True
                    ).start()
                streams_to_remove.append(stream_name)

        for stream_name in streams_to_remove:
            self.pending_reconnect.pop(stream_name, None)

    def monitor_streams(self):
        """
        Monitor stream health and restart dead processors
        Runs in a background thread
        """
        self.logger.info("Stream monitor started")

        while self.running:
            try:
                # Check MediaMTX health
                if not self.mediamtx.is_healthy():
                    self.logger.error("MediaMTX is not healthy!")
                    # Could implement restart logic here

                # Check processor health
                dead_processors = []
                for stream_name, proc in self.processors.items():
                    if not proc.is_alive():
                        self.logger.warning(f"Processor for {stream_name} died unexpectedly")
                        dead_processors.append(stream_name)

                # Clean up dead processors
                for stream_name in dead_processors:
                    self.stop_processor(stream_name)

                    # Check if stream is still active, restart processor
                    active_streams = self.mediamtx.get_active_streams()
                    if stream_name in active_streams:
                        self.logger.info(f"Stream {stream_name} is still active, restarting processor")
                        self.start_processor(stream_name)

                # Get active streams and ensure processors exist
                active_streams = self.mediamtx.get_active_streams()
                for stream_name in active_streams:
                    if stream_name not in self.processors and stream_name not in self.pending_reconnect:
                        self.logger.info(f"Found active stream without processor: {stream_name}")
                        self.start_processor(stream_name)

                # Check streams pending reconnection
                self._check_pending_reconnects()

            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")

            # Sleep before next check
            time.sleep(STREAM_HEALTH_CHECK_INTERVAL)

        self.logger.info("Stream monitor stopped")

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

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_streams, daemon=True)
        self.monitor_thread.start()

        # Suppress Flask/Werkzeug logging
        import logging as flask_logging
        flask_log = flask_logging.getLogger('werkzeug')
        flask_log.setLevel(flask_logging.ERROR)

        # Start Flask app in a separate thread
        flask_thread = threading.Thread(
            target=lambda: self.app.run(
                host=ORCHESTRATOR_HOST,
                port=ORCHESTRATOR_PORT,
                debug=False,
                use_reloader=False
            ),
            daemon=True
        )
        flask_thread.start()

        self.logger.info(f"Orchestrator API listening on {ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}")
        self.logger.info(f"Using processor class: {self.processor_class.__name__}")
        self.logger.info(f"Max concurrent processors: {MAX_PROCESSORS}")
        self.logger.info("=" * 60)
        self.logger.info("Ready to receive streams!")
        self.logger.info("Raspberry Pis can now connect via RTSP")
        self.logger.info("=" * 60)

        return True

    def stop(self):
        """Stop the orchestrator"""
        self.logger.info("Stopping orchestrator...")
        self.running = False

        # Set all devices offline (in parallel for speed)
        threads = []
        for stream_name, api_client in list(self.api_clients.items()):
            thread = threading.Thread(target=api_client.update_status, args=('offline',), daemon=False)
            thread.start()
            threads.append(thread)

        # Stop all processors
        for stream_name in list(self.processors.keys()):
            self.stop_processor(stream_name)

        # Wait for all status updates to complete (max 2 seconds)
        for thread in threads:
            thread.join(timeout=2)

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
