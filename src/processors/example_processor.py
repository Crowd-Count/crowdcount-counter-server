"""
Example Stream Processor
Demonstrates how to create a custom stream processor
"""

import cv2
import numpy as np
from .base_processor import BaseStreamProcessor


class ExampleProcessor(BaseStreamProcessor):
    """
    Example processor that adds a simple overlay to frames.
    Replace this with your actual computer vision logic.
    """

    def __init__(self, stream_name: str, rtsp_url: str, device_info: dict = None):
        super().__init__(stream_name, rtsp_url, device_info)
        self.detection_count = 0

    def on_start(self):
        """Initialize any resources needed for processing"""
        self.logger.info("Example processor starting")
        self.logger.info(f"Device ID: {self.device_id}")
        self.logger.info(f"Has password: {bool(self.password)}")
        self.logger.info(
            "This is where you'd load ML models, initialize detectors, etc."
        )
        # Example: Load a face detector, person detector, etc.
        # self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Note: You can use self.device_id and self.password for authentication or device-specific processing

    def process_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> np.ndarray:
        """
        Process each frame - this is where your computer vision logic goes

        Args:
            frame: BGR image from the video stream
            frame_number: Sequential frame number
            timestamp: Current timestamp

        Returns:
            Processed frame (for visualization/debugging)
        """
        # Make a copy so we don't modify the original
        display_frame = frame.copy()

        # Example processing: Simple edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Count non-zero pixels (just as an example metric)
        edge_pixels = np.count_nonzero(edges)

        # Example: Detect something (replace with your actual detection logic)
        # For demo purposes, we'll just randomly "detect" something
        if edge_pixels > 10000:  # Arbitrary threshold
            self.detection_count += 1

        # Add overlay with stats
        cv2.putText(
            display_frame,
            f"Device: {self.device_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            f"Frame: {frame_number}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            f"Detections: {self.detection_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Example: Draw edge overlay
        # display_frame[edges > 0] = [0, 255, 0]  # Highlight edges in green

        # YOUR COMPUTER VISION LOGIC GOES HERE
        # Examples:
        # - Person detection and counting
        # - Object tracking
        # - Action recognition
        # - Anomaly detection
        # - Face recognition
        # etc.

        return display_frame

    def on_stop(self):
        """Cleanup any resources"""
        self.logger.info(
            f"Example processor stopping. Total detections: {self.detection_count}"
        )
        # Example: Save results to database, close connections, etc.


class PassthroughProcessor(BaseStreamProcessor):
    """
    Simple passthrough processor that just logs frames.
    Useful for testing without heavy processing.
    """

    def on_start(self):
        self.logger.info("Passthrough processor started - minimal processing")

    def process_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> None:
        """Just log every 100th frame"""
        if frame_number % 100 == 0:
            self.logger.debug(f"Frame {frame_number}: {frame.shape}")
        return None  # No visualization needed

    def on_stop(self):
        self.logger.info("Passthrough processor stopped")


# You can add more processor types here:
# - CrowdCountProcessor: Counts people in the frame
# - FaceRecognitionProcessor: Identifies faces
# - MotionDetectionProcessor: Detects movement
# - ObjectTrackingProcessor: Tracks specific objects
# etc.
