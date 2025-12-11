"""
Display Stream Processor
Shows the video stream in a window for monitoring/debugging
"""

import cv2
import numpy as np
from .base_processor import BaseStreamProcessor


class DisplayProcessor(BaseStreamProcessor):
    """
    Processor that displays the stream in an OpenCV window
    Useful for monitoring and debugging streams
    """

    def __init__(self, stream_name: str, rtsp_url: str, device_info: dict = None):
        super().__init__(stream_name, rtsp_url)
        self.device_info = device_info or {}
        self.window_name = f"Stream: {stream_name}"

    def on_start(self):
        """Initialize the display window"""
        self.logger.info("Display processor starting")
        self.logger.info(f"Opening window: {self.window_name}")

        # Create a named window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

    def process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> np.ndarray:
        """
        Display each frame in a window

        Args:
            frame: BGR image from the video stream
            frame_number: Sequential frame number
            timestamp: Current timestamp

        Returns:
            The frame (for potential other uses)
        """
        # Create display frame with info overlay
        display_frame = frame.copy()

        # Get device info
        device_id = self.device_info.get('device_id', self.stream_name)

        # Add info overlay
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Dark semi-transparent background for text
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)

        # Text info
        cv2.putText(display_frame, f"Device: {device_id}",
                   (10, 25), font, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Stream: {self.stream_name}",
                   (10, 50), font, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Frame: {frame_number}",
                   (10, 75), font, 0.6, (0, 255, 0), 2)

        # FPS calculation
        if hasattr(self, 'start_time') and self.start_time:
            elapsed = timestamp - self.start_time
            fps = self.frames_processed / elapsed if elapsed > 0 else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}",
                       (10, 100), font, 0.6, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow(self.window_name, display_frame)

        # Wait 1ms for key press (required for cv2.imshow to work)
        key = cv2.waitKey(1) & 0xFF

        # Press 'q' to quit (will stop this processor)
        if key == ord('q'):
            self.logger.info("User pressed 'q', stopping processor")
            self.stop()

        return display_frame

    def on_stop(self):
        """Cleanup the display window"""
        self.logger.info("Display processor stopping")

        # Close the window
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)  # Process final events
        except Exception:
            pass

        self.logger.info(f"Window closed for {self.stream_name}")


class MultiStreamDisplayProcessor(BaseStreamProcessor):
    """
    Lightweight processor for multiple streams
    Shows stream in a smaller window suitable for monitoring many streams
    """

    def __init__(self, stream_name: str, rtsp_url: str, device_info: dict = None):
        super().__init__(stream_name, rtsp_url)
        self.device_info = device_info or {}
        self.window_name = f"{stream_name}"

    def on_start(self):
        """Initialize the display window"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 400, 300)

    def process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> np.ndarray:
        """Display frame with minimal overlay"""
        display_frame = frame.copy()

        # Simple device ID overlay
        device_id = self.device_info.get('device_id', self.stream_name)
        cv2.putText(display_frame, device_id, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(self.window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.stop()

        return display_frame

    def on_stop(self):
        """Cleanup"""
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except Exception:
            pass
