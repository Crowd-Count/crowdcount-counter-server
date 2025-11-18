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
from flask import Flask, request, jsonify
import multiprocessing

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
                self.logger.info(f"Received stream ready notification: {stream_name}")
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
                'max_processors': MAX_PROCESSORS
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

        self.logger.info(f"Starting processor for stream: {stream_name}")

        try:
            # Create processor instance
            processor = self.processor_class(stream_name, rtsp_url)

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
            del self.processors[stream_name]

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

        # Stop all processors
        for stream_name in list(self.processors.keys()):
            self.stop_processor(stream_name)

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
