"""
MediaMTX Manager
Handles starting, stopping, and monitoring the MediaMTX server
"""

import subprocess
import time
import logging
import requests
from pathlib import Path
from typing import Optional, List, Dict
import signal
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.server_config import MEDIAMTX_BINARY, MEDIAMTX_CONFIG, MEDIAMTX_API_URL


class MediaMTXManager:
    """Manages the MediaMTX server process"""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.running = False

        # Setup logging
        self.logger = logging.getLogger("MediaMTX")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def start(self) -> bool:
        """
        Start the MediaMTX server

        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            self.logger.warning("MediaMTX is already running")
            return True

        # Check if binary exists
        if not MEDIAMTX_BINARY.exists():
            self.logger.error(f"MediaMTX binary not found at {MEDIAMTX_BINARY}")
            self.logger.error("Run the setup script first: ./scripts/setup.sh")
            return False

        # Check if config exists
        if not MEDIAMTX_CONFIG.exists():
            self.logger.error(f"MediaMTX config not found at {MEDIAMTX_CONFIG}")
            return False

        try:
            self.logger.info("Starting MediaMTX server...")
            self.logger.info(f"Binary: {MEDIAMTX_BINARY}")
            self.logger.info(f"Config: {MEDIAMTX_CONFIG}")

            # Start MediaMTX process
            self.process = subprocess.Popen(
                [str(MEDIAMTX_BINARY), str(MEDIAMTX_CONFIG)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=MEDIAMTX_BINARY.parent.parent  # Run from base directory
            )

            # Wait a bit for server to start
            time.sleep(2)

            # Check if process is still running
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                self.logger.error("MediaMTX failed to start")
                self.logger.error(f"stdout: {stdout.decode()}")
                self.logger.error(f"stderr: {stderr.decode()}")
                return False

            # Verify server is responding
            if self._wait_for_ready(timeout=10):
                self.running = True
                self.logger.info("MediaMTX server started successfully")
                return True
            else:
                self.logger.error("MediaMTX server started but is not responding")
                self.stop()
                return False

        except Exception as e:
            self.logger.error(f"Error starting MediaMTX: {e}")
            return False

    def stop(self):
        """Stop the MediaMTX server"""
        if not self.running or not self.process:
            self.logger.info("MediaMTX is not running")
            return

        self.logger.info("Stopping MediaMTX server...")

        try:
            # Send SIGTERM
            self.process.terminate()

            # Wait up to 5 seconds for graceful shutdown
            try:
                self.process.wait(timeout=5)
                self.logger.info("MediaMTX stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop
                self.logger.warning("MediaMTX didn't stop gracefully, forcing kill")
                self.process.kill()
                self.process.wait()
                self.logger.info("MediaMTX killed")

        except Exception as e:
            self.logger.error(f"Error stopping MediaMTX: {e}")

        finally:
            self.process = None
            self.running = False

    def _wait_for_ready(self, timeout: int = 10) -> bool:
        """
        Wait for MediaMTX to be ready (check if process is running and ports are open)

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if ready, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if process is still running
            if self.process and self.process.poll() is None:
                # Process is running, try to connect to API port (even if API doesn't respond correctly)
                try:
                    # Just try to connect to the port, don't care about the response
                    response = requests.get(f"{MEDIAMTX_API_URL}/", timeout=3)
                    # If we get any response (even 404), the server is up
                    return True
                except requests.exceptions.ConnectionError:
                    # Port not open yet, keep waiting
                    pass
                except requests.exceptions.RequestException:
                    # Got some response, server is up
                    return True

            time.sleep(0.5)

        return False

    def is_healthy(self) -> bool:
        """
        Check if MediaMTX is healthy

        Returns:
            True if healthy, False otherwise
        """
        if not self.running or not self.process:
            return False

        # Check if process is still running
        if self.process.poll() is not None:
            self.logger.warning("MediaMTX process has died")
            self.running = False
            return False

        # Check if API port is responding (accept any response, even 404)
        try:
            response = requests.get(f"{MEDIAMTX_API_URL}/", timeout=5)
            # If we get any response (even 404), the server is up
            return True
        except requests.exceptions.ConnectionError:
            # Port not open
            return False
        except requests.exceptions.RequestException:
            # Got some response, server is up
            return True

    def get_paths(self) -> List[Dict]:
        """
        Get list of active paths (streams) from MediaMTX

        Returns:
            List of path dictionaries
        """
        try:
            response = requests.get(f"{MEDIAMTX_API_URL}/v3/paths/list", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('items', [])
            return []
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Error getting paths: {e}")
            return []

    def get_active_streams(self) -> List[str]:
        """
        Get list of currently active stream names

        Returns:
            List of stream names
        """
        paths = self.get_paths()
        active_streams = []

        for path in paths:
            # Check if path has readers (is actually streaming)
            if path.get('ready', False):
                active_streams.append(path.get('name', ''))

        return [s for s in active_streams if s]  # Filter out empty names

    def get_stream_info(self, stream_name: str) -> Optional[Dict]:
        """
        Get information about a specific stream

        Args:
            stream_name: Name of the stream

        Returns:
            Dictionary with stream info, or None if not found
        """
        try:
            response = requests.get(f"{MEDIAMTX_API_URL}/v3/paths/get/{stream_name}", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Error getting stream info for {stream_name}: {e}")
            return None


def main():
    """Test the MediaMTX manager"""
    manager = MediaMTXManager()

    def signal_handler(sig, frame):
        print("\nShutting down...")
        manager.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if manager.start():
        print("MediaMTX is running. Press Ctrl+C to stop.")
        try:
            while True:
                if not manager.is_healthy():
                    print("MediaMTX is not healthy!")
                    break

                streams = manager.get_active_streams()
                if streams:
                    print(f"Active streams: {', '.join(streams)}")

                time.sleep(5)
        finally:
            manager.stop()
    else:
        print("Failed to start MediaMTX")


if __name__ == "__main__":
    main()
