"""
MediaMTX Manager
Handles starting, stopping, and monitoring the MediaMTX server
"""

import subprocess
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import signal
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.server_config import MEDIAMTX_BINARY, MEDIAMTX_CONFIG, MEDIAMTX_API_URL

# Configuration constants
DEFAULT_TIMEOUT = 10  # Increased from 5s to 10s
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.5  # Exponential backoff: 0.5s, 1s, 2s
CONSECUTIVE_FAILURES_THRESHOLD = 5  # Restart MediaMTX after this many consecutive failures


class MediaMTXManager:
    """Manages the MediaMTX server process"""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.running = False

        # Connection pooling with requests.Session
        self._session = requests.Session()

        # Configure retry strategy for the session
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=5, pool_maxsize=10)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # Track consecutive API failures for health monitoring
        self._consecutive_failures = 0
        self._last_successful_api_call = time.time()

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
                    response = self._session.get(f"{MEDIAMTX_API_URL}/", timeout=3)
                    # If we get any response (even 404), the server is up
                    self._reset_failure_count()
                    return True
                except requests.exceptions.ConnectionError:
                    # Port not open yet, keep waiting
                    pass
                except requests.exceptions.RequestException:
                    # Got some response, server is up
                    self._reset_failure_count()
                    return True

            time.sleep(0.5)

        return False

    def _reset_failure_count(self):
        """Reset consecutive failure counter on successful API call"""
        self._consecutive_failures = 0
        self._last_successful_api_call = time.time()

    def _record_failure(self):
        """Record an API failure and check if restart is needed"""
        self._consecutive_failures += 1
        if self._consecutive_failures >= CONSECUTIVE_FAILURES_THRESHOLD:
            self.logger.warning(
                f"MediaMTX API failed {self._consecutive_failures} consecutive times - triggering restart"
            )
            return True
        return False

    def get_consecutive_failures(self) -> int:
        """Get the current consecutive failure count"""
        return self._consecutive_failures

    def needs_restart(self) -> bool:
        """Check if MediaMTX needs to be restarted due to API failures"""
        return self._consecutive_failures >= CONSECUTIVE_FAILURES_THRESHOLD

    def _make_api_request(self, endpoint: str, timeout: int = DEFAULT_TIMEOUT) -> Tuple[bool, Optional[requests.Response]]:
        """
        Make an API request with retry logic and failure tracking.

        Args:
            endpoint: API endpoint (e.g., "/v3/paths/list")
            timeout: Request timeout in seconds

        Returns:
            Tuple of (success, response)
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = self._session.get(
                    f"{MEDIAMTX_API_URL}{endpoint}",
                    timeout=timeout
                )
                self._reset_failure_count()
                return True, response
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = BACKOFF_FACTOR * (2 ** attempt)
                    self.logger.debug(f"API request failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    self.logger.warning(f"API request failed after {MAX_RETRIES} attempts: {e}")
                    self._record_failure()
                    return False, None
        return False, None

    def is_healthy(self) -> bool:
        """
        Check if MediaMTX is healthy.

        NOTE: This method does NOT reset the failure counter to avoid
        masking failures in get_paths(). Only get_paths() success resets
        the counter, ensuring auto-restart triggers when the paths API
        is unresponsive even if the root endpoint responds.

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
        # NOTE: We intentionally do NOT reset failure count here - only
        # successful get_paths() calls should reset it
        try:
            response = self._session.get(f"{MEDIAMTX_API_URL}/", timeout=DEFAULT_TIMEOUT)
            # If we get any response (even 404), the server is up
            return True
        except requests.exceptions.ConnectionError:
            # Port not open - but don't record failure here, let get_paths handle it
            return False
        except requests.exceptions.RequestException:
            # Got some response, server is up
            return True

    def get_paths(self) -> List[Dict]:
        """
        Get list of active paths (streams) from MediaMTX with retry logic.

        Returns:
            List of path dictionaries
        """
        success, response = self._make_api_request("/v3/paths/list")
        if success and response and response.status_code == 200:
            try:
                data = response.json()
                return data.get('items', [])
            except ValueError as e:
                self.logger.warning(f"Error parsing paths response: {e}")
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
        Get information about a specific stream with retry logic.

        Args:
            stream_name: Name of the stream

        Returns:
            Dictionary with stream info, or None if not found
        """
        success, response = self._make_api_request(f"/v3/paths/get/{stream_name}")
        if success and response and response.status_code == 200:
            try:
                return response.json()
            except ValueError as e:
                self.logger.warning(f"Error parsing stream info for {stream_name}: {e}")
        return None

    def restart(self) -> bool:
        """
        Restart the MediaMTX server.
        Used when API becomes unresponsive.

        Returns:
            True if restart was successful, False otherwise
        """
        self.logger.info("Restarting MediaMTX server...")
        self.stop()
        time.sleep(1)  # Brief pause before restart
        success = self.start()
        if success:
            self.logger.info("MediaMTX server restarted successfully")
            self._reset_failure_count()
        else:
            self.logger.error("Failed to restart MediaMTX server")
        return success


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
