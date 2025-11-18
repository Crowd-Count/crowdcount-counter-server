"""
Base Stream Processor
Abstract base class for all stream processors
"""

import cv2
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseStreamProcessor(ABC):
    """
    Base class for stream processors.
    Each processor handles one video stream and runs its own processing logic.
    """

    def __init__(self, stream_name: str, rtsp_url: str):
        """
        Initialize the stream processor

        Args:
            stream_name: Unique identifier for this stream (e.g., 'camera1')
            rtsp_url: Full RTSP URL to pull frames from (e.g., 'rtsp://127.0.0.1:8554/camera1')
        """
        self.stream_name = stream_name
        self.rtsp_url = rtsp_url
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None

        # Setup logging
        self.logger = logging.getLogger(f"Processor-{stream_name}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'%(asctime)s - {stream_name} - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Stats
        self.frames_processed = 0
        self.start_time = None
        self.last_frame_time = time.time()

    def connect_stream(self) -> bool:
        """
        Connect to the RTSP stream

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"Connecting to stream: {self.rtsp_url}")
            self.cap = cv2.VideoCapture(self.rtsp_url)

            # Set buffer size to minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                self.logger.error("Failed to open stream")
                return False

            # Test read a frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.logger.error("Failed to read initial frame")
                return False

            self.logger.info(f"Successfully connected to stream (resolution: {frame.shape[1]}x{frame.shape[0]})")
            return True

        except Exception as e:
            self.logger.error(f"Error connecting to stream: {e}")
            return False

    def disconnect_stream(self):
        """Disconnect from the RTSP stream"""
        if self.cap:
            self.cap.release()
            self.cap = None
            self.logger.info("Disconnected from stream")

    @abstractmethod
    def process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> Optional[np.ndarray]:
        """
        Process a single frame. Override this method with your custom logic.

        Args:
            frame: The video frame as a numpy array (BGR format)
            frame_number: Sequential frame number
            timestamp: Current timestamp

        Returns:
            Processed frame (optional, can return None if no visualization needed)
        """
        pass

    @abstractmethod
    def on_start(self):
        """
        Called when the processor starts.
        Override to perform initialization tasks.
        """
        pass

    @abstractmethod
    def on_stop(self):
        """
        Called when the processor stops.
        Override to perform cleanup tasks.
        """
        pass

    def run(self):
        """
        Main processing loop.
        Continuously reads frames from the stream and processes them.
        """
        self.logger.info("Starting processor")
        self.running = True
        self.start_time = time.time()

        # Connect to stream
        if not self.connect_stream():
            self.logger.error("Failed to connect to stream, exiting")
            self.running = False
            return

        # Call custom start logic
        try:
            self.on_start()
        except Exception as e:
            self.logger.error(f"Error in on_start: {e}")

        # Main processing loop
        frame_number = 0
        consecutive_failures = 0
        max_consecutive_failures = 30

        try:
            while self.running:
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    self.logger.warning(f"Failed to read frame ({consecutive_failures}/{max_consecutive_failures})")

                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error("Too many consecutive frame read failures, stopping")
                        break

                    time.sleep(0.1)
                    continue

                # Reset failure counter on successful read
                consecutive_failures = 0
                frame_number += 1
                self.frames_processed += 1
                timestamp = time.time()
                self.last_frame_time = timestamp

                # Process the frame (custom logic)
                try:
                    processed_frame = self.process_frame(frame, frame_number, timestamp)
                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_number}: {e}")
                    continue

                # Log stats periodically (every 300 frames = ~10 seconds at 30fps)
                if frame_number % 300 == 0:
                    elapsed = timestamp - self.start_time
                    fps = self.frames_processed / elapsed if elapsed > 0 else 0
                    self.logger.info(f"Processed {self.frames_processed} frames ({fps:.1f} fps)")

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Error in processing loop: {e}")
        finally:
            # Cleanup
            self.running = False
            elapsed = time.time() - self.start_time if self.start_time else 0

            try:
                self.on_stop()
            except Exception as e:
                self.logger.error(f"Error in on_stop: {e}")

            self.disconnect_stream()
            self.logger.info(f"Processor stopped. Processed {self.frames_processed} frames in {elapsed:.1f}s")

    def stop(self):
        """Stop the processor"""
        self.logger.info("Stop requested")
        self.running = False

    def get_stats(self) -> dict:
        """
        Get current processor statistics

        Returns:
            Dictionary with processor stats
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frames_processed / elapsed if elapsed > 0 else 0

        return {
            'stream_name': self.stream_name,
            'rtsp_url': self.rtsp_url,
            'running': self.running,
            'frames_processed': self.frames_processed,
            'elapsed_time': elapsed,
            'fps': fps,
            'last_frame_time': self.last_frame_time
        }
