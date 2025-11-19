"""
Stream Orchestrator
Main orchestrator that manages MediaMTX and stream processors
Automatically spawns a processor for each new stream
"""

import sys
import time
import logging
import threading
import signal
from pathlib import Path
from typing import Dict, Type
from flask import Flask, request, jsonify, Response, render_template_string
import multiprocessing
import cv2
import queue

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.crowdcount_api_client import CrowdCountAPIClient
from config.server_config import (
    ORCHESTRATOR_HOST,
    ORCHESTRATOR_PORT,
    MAX_PROCESSORS,
    STREAM_HEALTH_CHECK_INTERVAL,
    RTSP_HOST,
    RTSP_PORT,
    LOG_LEVEL
)
from streaming.mediamtx_manager import MediaMTXManager
from processors import BaseStreamProcessor, ExampleProcessor


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

        # Track API clients for device status updates
        self.api_clients: Dict[str, CrowdCountAPIClient] = {}

        # Track device information for each stream (device_id, password, etc.)
        self.device_info: Dict[str, dict] = {}

        self.running = False
        self.monitor_thread = None

        # Setup logging
        self.logger = logging.getLogger("Orchestrator")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, LOG_LEVEL))

        # Flask app for receiving MediaMTX webhooks
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes for MediaMTX webhooks"""

        @self.app.route('/stream/ready', methods=['POST'])
        def stream_ready():
            """Called by MediaMTX when a stream becomes ready"""
            data = request.get_json()
            stream_name = data.get('stream', '')

            if stream_name:
                # Check if this stream is already being processed (avoid duplicate calls)
                if stream_name in self.processors and self.processors[stream_name].is_alive():
                    # Stream already has a processor running, ignore
                    return jsonify({'status': 'ok', 'message': 'Already processing'}), 200

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

                # Detailed logging for testing
                self.logger.info("=" * 60)
                self.logger.info(f"🔗 NEW DEVICE CONNECTED")
                self.logger.info(f"   Stream Name: {stream_name}")
                self.logger.info(f"   Device ID:   {device_id}")
                self.logger.info(f"   Password:    {password if password else '(none)'}")
                self.logger.info("=" * 60)

                self.start_processor(stream_name)
                return jsonify({'status': 'ok'}), 200

            return jsonify({'status': 'error', 'message': 'No stream name provided'}), 400

        @self.app.route('/stream/stopped', methods=['POST'])
        def stream_stopped():
            """Called by MediaMTX when a stream stops"""
            data = request.get_json()
            stream_name = data.get('stream', '')

            if stream_name:
                self.logger.info(f"Received stream stopped notification: {stream_name}")
                self.stop_processor(stream_name)

                # Clean up device info for this stream
                if stream_name in self.device_info:
                    del self.device_info[stream_name]

                return jsonify({'status': 'ok'}), 200

            return jsonify({'status': 'error', 'message': 'No stream name provided'}), 400

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
            Useful for pre-registering device info before the stream connects.
            """
            data = request.get_json()
            stream_name = data.get('stream_name', '')
            device_id = data.get('device_id', stream_name)
            password = data.get('password', '')

            if not stream_name:
                return jsonify({'status': 'error', 'message': 'stream_name is required'}), 400

            # Store device info
            self.device_info[stream_name] = {
                'device_id': device_id,
                'password': password,
                'timestamp': time.time()
            }

            self.logger.info(f"Registered device info for stream: {stream_name} (device_id: {device_id})")
            return jsonify({'status': 'ok', 'stream_name': stream_name, 'device_id': device_id}), 200

        @self.app.route('/view/<stream_name>')
        def view_stream(stream_name):
            """Serve HTML viewer for a stream"""
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>CrowdCount - Stream {{ stream_name }}</title>
                <style>
                    body {
                        margin: 0;
                        padding: 20px;
                        background: #1a1a1a;
                        font-family: Arial, sans-serif;
                        color: #fff;
                    }
                    .container {
                        max-width: 1400px;
                        margin: 0 auto;
                    }
                    h1 {
                        text-align: center;
                        color: #00ff00;
                    }
                    .video-container {
                        text-align: center;
                        margin: 20px 0;
                    }
                    img {
                        max-width: 100%;
                        border: 2px solid #00ff00;
                        border-radius: 8px;
                    }
                    .info {
                        text-align: center;
                        margin: 10px 0;
                        color: #aaa;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎥 CrowdCount Stream Viewer</h1>
                    <div class="info">Stream: {{ stream_name }}</div>
                    <div class="video-container">
                        <img src="/video_feed/{{ stream_name }}" alt="Live Stream">
                    </div>
                    <div class="info">Live YOLO Detection Feed</div>
                </div>
            </body>
            </html>
            """
            return render_template_string(html, stream_name=stream_name)

        @self.app.route('/video_feed/<stream_name>')
        def video_feed(stream_name):
            """Serve MJPEG video feed for a stream"""
            def generate():
                if stream_name not in self.frame_queues:
                    return

                frame_queue = self.frame_queues[stream_name]

                while True:
                    try:
                        # Get frame from queue with timeout
                        frame = frame_queue.get(timeout=1.0)

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
            # Clean up device info
            if stream_name in self.device_info:
                del self.device_info[stream_name]

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
                    if stream_name not in self.processors:
                        self.logger.info(f"Found active stream without processor: {stream_name}")
                        self.start_processor(stream_name)

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
