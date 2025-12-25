"""
People Counter Processor - Method 3: Dual-Line Gate
Enhanced overhead doorway person counter using YOLO with ByteTrack tracking

METHOD 3: Dual-Line Gate (Two Parallel Lines)
- Two parallel lines (LINE_A and LINE_B) separated by a configurable gap
- Direction determined by which line is crossed first
- Requires crossing both lines in sequence to count
- Robust: impossible to count without crossing both lines
- Resistant to on-line oscillation
"""

import cv2
import numpy as np
import time
import math
import threading
import requests
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
import os

from .base_processor import BaseStreamProcessor

# Import API client
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from api.crowdcount_api_client import CrowdCountAPIClient

# Get project root directory (crowdcount-counter-server/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Try to import YOLO, provide helpful error if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

# Try to import torch for ReID feature extraction
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not installed. ReID features disabled.")

# Import deep ReID extractor
from .deep_reid import create_reid_extractor, TORCHREID_AVAILABLE


class CrossingDirection(Enum):
    """Direction of crossing the doorway line"""
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    NONE = "none"


@dataclass
class Detection:
    """Represents a single person detection with optional ReID features"""
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    timestamp: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None  # ByteTrack assigned ID
    appearance_features: Optional[np.ndarray] = None  # ReID embedding

    def __hash__(self):
        """Make Detection hashable for set operations"""
        return hash((self.center, self.timestamp))


class PersonTracker:
    """
    METHOD 3: Dual-Line Gate (Two Parallel Lines)

    Key features:
    - Two parallel lines (LINE_A at 45%, LINE_B at 55% of frame width)
    - Direction determined by which line is crossed first
    - Must cross both lines in sequence within time limit
    - Anti-bounce logic prevents oscillation from causing false counts
    """

    # Method 3 configuration constants
    DUAL_LINE_GAP = 0.10              # Gap between lines (10% of frame width)
    MIN_TRANSIT_TIME = 0.05           # Minimum seconds between line A and B crossings
    MAX_TRANSIT_TIME = 3.0            # Maximum seconds between line A and B crossings
    MAX_OSCILLATIONS = 3              # Max line crossings in short period before ignore
    OSCILLATION_WINDOW = 2.0          # Time window for oscillation detection

    def __init__(self, track_id: int, initial_detection: Detection):
        self.track_id = track_id
        self.detections = deque(maxlen=60)  # Increased for better trajectory analysis
        self.positions = deque(maxlen=60)
        self.velocities = deque(maxlen=15)
        self.creation_time = time.time()
        self.last_update_time = time.time()

        # Multiple crossing state tracking
        self.total_crossings = 0
        self.crossing_events = []  # List of (timestamp, direction) tuples
        self.last_side = None  # Track which side of center line person is on
        self.current_side = None
        self.last_crossing_time = 0
        self.min_crossing_interval = 0.5  # Increased: minimum seconds between crossings

        # Overall tracking stats
        self.peak_confidence = 0.0

        # Trajectory validation for counting
        self.side_history = deque(maxlen=10)  # Track recent side positions
        self.detection_count = 0  # Total detections for this track
        self.min_detections_to_count = 3  # Require at least N detections before counting
        self.consecutive_side_detections = 0  # Consecutive detections on same side
        self.min_consecutive_for_crossing = 1  # Reduced from 2 - trajectory validation handles consistency

        # ReID appearance features
        self.appearance_history = deque(maxlen=10)  # Store recent appearance embeddings
        self.mean_appearance = None  # Running mean of appearance features

        # Pending crossing validation
        self.pending_crossing = None  # (direction, timestamp) if crossing needs validation
        self.crossing_start_position = None  # Position when crossing started

        # METHOD 3: Dual-line gate state
        self.crossed_line_a = False           # Whether LINE_A has been crossed
        self.crossed_line_b = False           # Whether LINE_B has been crossed
        self.line_a_cross_time = None         # Timestamp when LINE_A was crossed
        self.line_b_cross_time = None         # Timestamp when LINE_B was crossed
        self.line_a_direction = None          # Direction when crossing LINE_A
        self.line_b_direction = None          # Direction when crossing LINE_B
        self.line_a_position = None           # Position when crossing LINE_A (for ReID checkpoint)
        self.previous_line_a_side = None      # Previous side relative to LINE_A
        self.previous_line_b_side = None      # Previous side relative to LINE_B
        self.line_crossing_history = deque(maxlen=10)  # Recent line crossings for anti-bounce

        # Add initial detection
        self.update(initial_detection)

    def update(self, detection: Detection):
        """Update tracker with new detection and ReID features"""
        self.detections.append(detection)
        self.positions.append(detection.center)
        self.last_update_time = time.time()
        self.peak_confidence = max(self.peak_confidence, detection.confidence)
        self.detection_count += 1

        # Calculate velocity
        if len(self.positions) >= 2:
            dx = self.positions[-1][0] - self.positions[-2][0]
            dy = self.positions[-1][1] - self.positions[-2][1]
            dt = detection.timestamp - self.detections[-2].timestamp if len(self.detections) >= 2 else 0.033
            if dt > 0:
                vx = dx / dt
                vy = dy / dt
                self.velocities.append((vx, vy))

        # Update ReID appearance features
        if detection.appearance_features is not None:
            self.appearance_history.append(detection.appearance_features)
            # Update running mean of appearance (re-normalize to maintain L2 norm)
            if len(self.appearance_history) > 0:
                self.mean_appearance = np.mean(list(self.appearance_history), axis=0)
                norm = np.linalg.norm(self.mean_appearance)
                if norm > 0:
                    self.mean_appearance = self.mean_appearance / norm

    def update_appearance(self, features: np.ndarray):
        """Update appearance features separately"""
        if features is not None:
            self.appearance_history.append(features)
            # Update running mean of appearance (re-normalize to maintain L2 norm)
            if len(self.appearance_history) > 0:
                self.mean_appearance = np.mean(list(self.appearance_history), axis=0)
                norm = np.linalg.norm(self.mean_appearance)
                if norm > 0:
                    self.mean_appearance = self.mean_appearance / norm

    def get_appearance_similarity(self, features: np.ndarray) -> float:
        """Calculate cosine similarity with this tracker's appearance.

        Note: Features are already L2-normalized, so cosine similarity = dot product.
        """
        if self.mean_appearance is None or features is None:
            return 0.0
        # For L2-normalized vectors, cosine similarity is just the dot product
        similarity = np.dot(self.mean_appearance, features)
        # Clamp to [0, 1] range (can be slightly out of range due to floating point)
        return max(0.0, min(1.0, float(similarity)))

    def _reset_gate_state(self):
        """METHOD 3: Reset the dual-line gate state."""
        self.crossed_line_a = False
        self.crossed_line_b = False
        self.line_a_cross_time = None
        self.line_b_cross_time = None
        self.line_a_direction = None
        self.line_b_direction = None
        self.line_a_position = None

    def _is_oscillating(self, current_time: float) -> bool:
        """METHOD 3: Check if track is oscillating (bouncing between lines)."""
        recent_crossings = [t for t in self.line_crossing_history
                           if current_time - t < self.OSCILLATION_WINDOW]
        return len(recent_crossings) >= self.MAX_OSCILLATIONS

    def check_crossing(self, frame_width: int, doorway_center: float, doorway_width: float,
                       dual_line_gap: float = None) -> Optional[CrossingDirection]:
        """
        METHOD 3: Dual-Line Gate (Two Parallel Lines)

        Two lines separated by a gap:
        - LINE_A at (center - gap/2)
        - LINE_B at (center + gap/2)

        A crossing is only counted when:
        1. Track crosses LINE_A (records direction and time)
        2. Track crosses LINE_B within time window
        3. Both crossings are in the same direction
        4. Transit time is within valid range

        Anti-bounce: Ignores tracks that oscillate too frequently.
        """
        if len(self.positions) < 2:
            return None

        current_x = self.positions[-1][0]
        previous_x = self.positions[-2][0] if len(self.positions) >= 2 else current_x
        current_time = time.time()

        # Use configured gap or default
        gap = dual_line_gap if dual_line_gap is not None else self.DUAL_LINE_GAP

        # Calculate line positions
        center_x = frame_width * doorway_center
        half_gap = frame_width * gap / 2
        line_a_x = center_x - half_gap  # Left line
        line_b_x = center_x + half_gap  # Right line

        # Determine current side relative to each line
        current_side_a = 'left' if current_x < line_a_x else 'right'
        current_side_b = 'left' if current_x < line_b_x else 'right'

        # Check for oscillation (anti-bounce)
        if self._is_oscillating(current_time):
            return None

        # Check for LINE_A crossing
        if self.previous_line_a_side is not None and current_side_a != self.previous_line_a_side:
            # Crossed LINE_A
            self.line_crossing_history.append(current_time)

            if self.previous_line_a_side == 'left' and current_side_a == 'right':
                self.line_a_direction = 'left_to_right'
            else:
                self.line_a_direction = 'right_to_left'

            self.crossed_line_a = True
            self.line_a_cross_time = current_time
            self.line_a_position = self.positions[-1]

        self.previous_line_a_side = current_side_a

        # Check for LINE_B crossing
        if self.previous_line_b_side is not None and current_side_b != self.previous_line_b_side:
            # Crossed LINE_B
            self.line_crossing_history.append(current_time)

            if self.previous_line_b_side == 'left' and current_side_b == 'right':
                self.line_b_direction = 'left_to_right'
            else:
                self.line_b_direction = 'right_to_left'

            self.crossed_line_b = True
            self.line_b_cross_time = current_time

        self.previous_line_b_side = current_side_b

        # Check if gate crossing is complete
        crossing_to_emit = None

        if self.crossed_line_a and self.crossed_line_b:
            # Both lines crossed - validate the crossing
            time_diff = abs(self.line_b_cross_time - self.line_a_cross_time)

            if self.MIN_TRANSIT_TIME <= time_diff <= self.MAX_TRANSIT_TIME:
                # Valid transit time
                if self.line_a_direction == self.line_b_direction:
                    # Consistent direction through gate

                    # Determine which line was crossed first
                    if self.line_a_cross_time < self.line_b_cross_time:
                        # Crossed A first, then B -> left to right
                        if self.line_a_direction == 'left_to_right':
                            if current_time - self.last_crossing_time >= self.min_crossing_interval:
                                crossing_to_emit = CrossingDirection.LEFT_TO_RIGHT
                                self.total_crossings += 1
                                self.crossing_events.append((current_time, crossing_to_emit))
                                self.last_crossing_time = current_time
                                self.last_side = 'right'
                                self.current_side = 'right'
                    else:
                        # Crossed B first, then A -> right to left
                        if self.line_b_direction == 'right_to_left':
                            if current_time - self.last_crossing_time >= self.min_crossing_interval:
                                crossing_to_emit = CrossingDirection.RIGHT_TO_LEFT
                                self.total_crossings += 1
                                self.crossing_events.append((current_time, crossing_to_emit))
                                self.last_crossing_time = current_time
                                self.last_side = 'left'
                                self.current_side = 'left'

            # Reset gate state after processing
            self._reset_gate_state()

        # Check for timeout on pending LINE_A crossing
        if self.crossed_line_a and not self.crossed_line_b:
            if self.line_a_cross_time and (current_time - self.line_a_cross_time > self.MAX_TRANSIT_TIME):
                # Timeout - reset gate
                self._reset_gate_state()

        # Update legacy side tracking for visualization compatibility
        if current_x < center_x:
            self.current_side = 'left'
        else:
            self.current_side = 'right'

        return crossing_to_emit

    def _validate_crossing(self, frame_width: int, doorway_center: float) -> bool:
        """Legacy validation method - always returns True for Method 3."""
        return True

    def get_smooth_velocity(self) -> Tuple[float, float]:
        """Get smoothed velocity estimate"""
        if not self.velocities:
            return (0.0, 0.0)

        # Weighted average with recent velocities having more weight
        weights = np.linspace(0.5, 1.0, len(self.velocities))
        vx_avg = np.average([v[0] for v in self.velocities], weights=weights)
        vy_avg = np.average([v[1] for v in self.velocities], weights=weights)

        return (vx_avg, vy_avg)

    def time_since_update(self) -> float:
        """Time since last update"""
        return time.time() - self.last_update_time

    def get_predicted_position(self, current_time: float = None) -> Tuple[int, int]:
        """
        Get predicted position based on velocity interpolation.
        Used to maintain tracking through dropped frames.

        Args:
            current_time: Current timestamp. If None, uses time.time()

        Returns:
            Predicted (x, y) position tuple
        """
        if not self.positions:
            return (0, 0)

        last_pos = self.positions[-1]

        if current_time is None:
            current_time = time.time()

        time_delta = current_time - self.last_update_time

        # Don't extrapolate too far into the future (max 1 second)
        time_delta = min(time_delta, 1.0)

        # Get velocity and predict new position
        vx, vy = self.get_smooth_velocity()

        predicted_x = last_pos[0] + vx * time_delta
        predicted_y = last_pos[1] + vy * time_delta

        return (int(predicted_x), int(predicted_y))

    def interpolate_positions(self, current_time: float, num_steps: int = 3) -> List[Tuple[int, int]]:
        """
        Generate interpolated positions between last known position and predicted current position.
        Useful for checking crossings that may have occurred during dropped frames.

        Args:
            current_time: Current timestamp
            num_steps: Number of intermediate positions to generate

        Returns:
            List of interpolated (x, y) positions
        """
        if not self.positions:
            return []

        last_pos = self.positions[-1]
        predicted_pos = self.get_predicted_position(current_time)

        # Generate intermediate positions
        positions = []
        for i in range(1, num_steps + 1):
            t = i / num_steps
            interp_x = int(last_pos[0] + (predicted_pos[0] - last_pos[0]) * t)
            interp_y = int(last_pos[1] + (predicted_pos[1] - last_pos[1]) * t)
            positions.append((interp_x, interp_y))

        return positions


class PeopleCounterProcessor(BaseStreamProcessor):
    """
    Overhead doorway person counter processor using YOLO
    Tracks people crossing a virtual line and maintains counts
    """

    def __init__(self, stream_name: str, rtsp_url: str, device_info: Optional[dict] = None, frame_queue=None):
        super().__init__(stream_name, rtsp_url, device_info, frame_queue)

        # YOLO model setup
        self.model = None
        self.model_path = str(MODELS_DIR / "yolo11s.pt")  # Using YOLOv11 Small model for better accuracy
        self.bytetrack_config = str(MODELS_DIR / "bytetrack.yaml")  # ByteTrack config for group tracking
        self.class_names = {0: 'person'}  # Default

        # Doorway configuration - TUNED FOR GROUP DETECTION
        self.doorway_config = {
            'doorway_center': 0.5,      # 50% of frame width (center line)
            'doorway_width': 0.02,      # 2% of frame width (narrow line)
            'approach_zone': 0.15,      # 15% approach zone on each side
            'min_path_length': 50,      # Minimum pixels traveled
            'max_tracking_distance': 100,  # TUNED: Reduced from 120 for tighter matching
            'tracker_timeout': 3.0,     # TUNED: Increased from 2.0 for slow-moving groups
            'yolo_conf_threshold': 0.20,  # TUNED: Lowered to catch detections in doorway zone
            'yolo_iou_threshold': 0.3,   # TUNED: Lowered from 0.45 to separate overlapping people
            'min_crossing_interval': 0.5,  # TUNED: Increased from 0.3 to prevent double counts
            'reid_similarity_threshold': 0.6,  # TUNED: Increased for deep ReID (more discriminative)
            'use_bytetrack': True,       # NEW: Enable ByteTrack
            'use_reid': True,            # NEW: Enable ReID features
            'reid_model': 'osnet_x0_25',  # Deep ReID model: osnet_x0_25 (fastest ~3ms GPU) - was osnet_x1_0
            # METHOD 3: Dual-Line Gate parameters
            'dual_line_gap': 0.10,          # Gap between lines (10% of frame width)
            'min_transit_time': 0.05,       # Min seconds between line crossings
            'max_transit_time': 3.0,        # Max seconds between line crossings
            'max_oscillations': 3,          # Max crossings before anti-bounce
            'oscillation_window': 2.0,      # Time window for oscillation detection
        }

        # Deep ReID feature extractor (initialized in on_start)
        self.reid_extractor = None

        # Tracking
        self.trackers: Dict[int, PersonTracker] = {}
        self.next_track_id = 1
        self.track_id_pool = []

        # Crossing counts
        self.crossings = {
            CrossingDirection.LEFT_TO_RIGHT: 0,
            CrossingDirection.RIGHT_TO_LEFT: 0
        }
        self.crossing_log = []

        # Initialize API client with device credentials
        # API is enabled only if device has credentials
        device_id = self.device_info.get('device_id', self.stream_name)
        password = self.device_info.get('password', '')
        self.api_client = CrowdCountAPIClient(
            device_id=device_id,
            password=password,
            base_url='https://api.crowdcount.tech',
            timeout=5
        )

        # Fetch direction flip setting and set up periodic refresh
        self.direction_flipped = self.api_client.get_direction_flipped()
        self.direction_refresh_timer = None
        self.logger.info(f"Direction flipped setting initialized: {self.direction_flipped}")

    def on_start(self):
        """Initialize YOLO model with ByteTrack when processor starts"""
        if not YOLO_AVAILABLE:
            print(f"ERROR: YOLO not available - cannot start people counter", flush=True)
            self.logger.error("YOLO not available - cannot start people counter")
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")

        # Start periodic refresh of direction_flipped setting (every 60 seconds)
        self._schedule_direction_refresh()

        try:
            import torch

            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            self.device = 'cuda' if cuda_available else 'cpu'

            print(f"Loading YOLO model: {self.model_path}", flush=True)
            print(f"Using device: {self.device}", flush=True)
            print(f"ByteTrack config: {self.bytetrack_config}", flush=True)
            if cuda_available:
                print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

            self.logger.info(f"Loading YOLO model: {self.model_path}")
            self.logger.info(f"Using device: {self.device}")
            self.logger.info(f"ByteTrack enabled: {self.doorway_config['use_bytetrack']}")
            self.logger.info(f"ReID enabled: {self.doorway_config['use_reid']}")

            # Load model with explicit CUDA device
            self.model = YOLO(self.model_path)
            self.model.to(self.device)

            # Verify device
            model_device = next(self.model.model.parameters()).device
            print(f"Model loaded on device: {model_device}", flush=True)
            self.logger.info(f"Model loaded on device: {model_device}")

            # Get class names from model
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"Model classes: {', '.join([f'{k}:{v}' for k, v in list(self.class_names.items())[:5]])}...", flush=True)
                self.logger.info(f"Model classes: {', '.join([f'{k}:{v}' for k, v in self.class_names.items()])}")

            # Log tuned parameters
            print(f"✓ Tuned parameters for group detection:", flush=True)
            print(f"  - Confidence threshold: {self.doorway_config['yolo_conf_threshold']}", flush=True)
            print(f"  - IOU threshold: {self.doorway_config['yolo_iou_threshold']}", flush=True)
            print(f"  - Min crossing interval: {self.doorway_config['min_crossing_interval']}s", flush=True)
            print(f"  - Tracker timeout: {self.doorway_config['tracker_timeout']}s", flush=True)

            print("✓ YOLO + ByteTrack loaded successfully with CUDA!" if cuda_available else "✓ YOLO + ByteTrack loaded (CPU mode)", flush=True)
            self.logger.info("YOLO model with ByteTrack loaded successfully")

            # Initialize Deep ReID extractor
            if self.doorway_config['use_reid']:
                print(f"Loading Deep ReID model: {self.doorway_config['reid_model']}...", flush=True)
                self.reid_extractor = create_reid_extractor(
                    model_name=self.doorway_config['reid_model'],
                    device=self.device,
                    verbose=True
                )
                print(f"✓ Deep ReID loaded (feature dim: {self.reid_extractor.feature_dim})", flush=True)
                self.logger.info(f"Deep ReID initialized: {self.doorway_config['reid_model']}")

        except Exception as e:
            print(f"ERROR: Failed to load YOLO model: {e}", flush=True)
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

    def on_stop(self):
        """Cleanup when processor stops"""
        # Cancel the direction refresh timer
        if self.direction_refresh_timer:
            self.direction_refresh_timer.cancel()

        # Print summary
        total = sum(self.crossings.values())
        lr_count = self.crossings[CrossingDirection.LEFT_TO_RIGHT]
        rl_count = self.crossings[CrossingDirection.RIGHT_TO_LEFT]
        net_flow = lr_count - rl_count

        self.logger.info("="*60)
        self.logger.info("PEOPLE COUNTER - Summary")
        self.logger.info("="*60)
        self.logger.info(f"Total Crossings: {total}")
        self.logger.info(f"  Left → Right: {lr_count}")
        self.logger.info(f"  Right → Left: {rl_count}")
        self.logger.info(f"  Net Flow (L→R): {net_flow:+d}")
        self.logger.info("="*60)

    def _schedule_direction_refresh(self):
        """Schedule the next direction_flipped setting refresh"""
        self.direction_refresh_timer = threading.Timer(60.0, self._refresh_direction_setting)
        self.direction_refresh_timer.daemon = True
        self.direction_refresh_timer.start()

    def _refresh_direction_setting(self):
        """Fetch the latest direction_flipped setting from API"""
        try:
            new_value = self.api_client.get_direction_flipped()
            if new_value != self.direction_flipped:
                self.logger.info(f"Direction flipped setting changed: {self.direction_flipped} → {new_value}")
                self.direction_flipped = new_value
        except Exception as e:
            self.logger.warning(f"Failed to refresh direction_flipped setting: {e}")

        # Schedule next refresh
        self._schedule_direction_refresh()

    def extract_detections(self, frame) -> List[Detection]:
        """Extract person detections from frame using YOLO with ByteTrack tracking"""
        detections = []

        try:
            if self.doorway_config['use_bytetrack']:
                # Use ByteTrack for improved tracking (especially for groups)
                results = self.model.track(
                    frame,
                    conf=self.doorway_config['yolo_conf_threshold'],
                    iou=self.doorway_config['yolo_iou_threshold'],
                    classes=[0],  # Only detect person class
                    persist=True,  # Persist tracks across frames
                    tracker=self.bytetrack_config,  # Use our custom ByteTrack config
                    verbose=False
                )
            else:
                # Fallback to standard detection
                results = self.model(
                    frame,
                    conf=self.doorway_config['yolo_conf_threshold'],
                    iou=self.doorway_config['yolo_iou_threshold'],
                    classes=[0],
                    verbose=False
                )

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    for i, box in enumerate(result.boxes):
                        # Get confidence and class
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.class_names.get(class_id, 'unknown')

                        # ONLY process person detections (class_id == 0)
                        if class_id != 0:
                            continue

                        # Get ByteTrack ID if available
                        track_id = None
                        if hasattr(box, 'id') and box.id is not None:
                            track_id = int(box.id[0])

                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Calculate center and dimensions
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        width = int(x2 - x1)
                        height = int(y2 - y1)

                        # Create detection WITHOUT ReID features
                        # ReID is extracted later only for unmatched detections (performance optimization)
                        detection = Detection(
                            center=(center_x, center_y),
                            bbox=(int(x1), int(y1), width, height),
                            confidence=confidence,
                            timestamp=time.time(),
                            class_id=class_id,
                            class_name=class_name,
                            track_id=track_id,
                            appearance_features=None  # Extracted on-demand in update_trackers
                        )
                        detections.append(detection)

        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

        return detections

    def _extract_appearance_features(self, frame, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        """
        Extract deep ReID appearance features from a person bounding box.
        Uses OSNet pretrained on person re-identification datasets.
        Returns 512-dimensional L2-normalized feature vector.
        """
        if self.reid_extractor is None:
            return None

        return self.reid_extractor.extract(frame, x1, y1, x2, y2)

    def _check_and_handle_crossing(self, track_id: int, detection: Detection, frame_width: int, current_time: float):
        """Check for crossing and handle API calls if crossing detected."""
        crossing_dir = self.trackers[track_id].check_crossing(
            frame_width,
            self.doorway_config['doorway_center'],
            self.doorway_config['doorway_width'],
            self.doorway_config['dual_line_gap']  # METHOD 3: pass dual_line_gap for gate width
        )

        if crossing_dir and crossing_dir != CrossingDirection.NONE:
            # A validated crossing occurred!
            self.crossings[crossing_dir] += 1

            # Make API call with device credentials
            # Apply direction flip logic if enabled
            if self.direction_flipped:
                if crossing_dir == CrossingDirection.LEFT_TO_RIGHT:
                    threading.Thread(target=self.api_client.subtract_from_count, args=(1,), daemon=True).start()
                elif crossing_dir == CrossingDirection.RIGHT_TO_LEFT:
                    threading.Thread(target=self.api_client.add_to_count, args=(1,), daemon=True).start()
            else:
                if crossing_dir == CrossingDirection.LEFT_TO_RIGHT:
                    threading.Thread(target=self.api_client.add_to_count, args=(1,), daemon=True).start()
                elif crossing_dir == CrossingDirection.RIGHT_TO_LEFT:
                    threading.Thread(target=self.api_client.subtract_from_count, args=(1,), daemon=True).start()

            # Log the crossing
            crossing_info = {
                'timestamp': current_time,
                'track_id': track_id,
                'direction': crossing_dir.value,
                'confidence': self.trackers[track_id].peak_confidence,
                'total_crossings_by_person': self.trackers[track_id].total_crossings,
                'position': detection.center,
                'detection_count': self.trackers[track_id].detection_count
            }
            self.crossing_log.append(crossing_info)

            total = sum(self.crossings.values())
            self.logger.info(f"CROSSING #{total}: {crossing_dir.value} | "
                           f"Track #{track_id} (detections: {self.trackers[track_id].detection_count}) | "
                           f"L->R: {self.crossings[CrossingDirection.LEFT_TO_RIGHT]} | "
                           f"R->L: {self.crossings[CrossingDirection.RIGHT_TO_LEFT]}")

    def _match_by_position_and_reid(self, detection: Detection) -> Optional[int]:
        """
        Match detection to tracker using position distance and ReID similarity.
        Used for detections that ByteTrack couldn't match directly.
        """
        if not self.trackers:
            return None

        best_match_id = None
        best_score = float('inf')

        for track_id, tracker in self.trackers.items():
            if len(tracker.positions) == 0:
                continue

            # Calculate distance to last known position
            last_pos = tracker.positions[-1]
            dist = math.sqrt(
                (detection.center[0] - last_pos[0])**2 +
                (detection.center[1] - last_pos[1])**2
            )

            # Consider velocity for prediction
            vx, vy = tracker.get_smooth_velocity()
            time_diff = detection.timestamp - tracker.last_update_time

            predicted_x = last_pos[0] + vx * time_diff
            predicted_y = last_pos[1] + vy * time_diff

            pred_dist = math.sqrt(
                (detection.center[0] - predicted_x)**2 +
                (detection.center[1] - predicted_y)**2
            )

            position_dist = min(dist, pred_dist)

            if position_dist > self.doorway_config['max_tracking_distance']:
                continue

            # Calculate ReID similarity score (if available)
            reid_score = 0.0
            if (self.doorway_config['use_reid'] and
                detection.appearance_features is not None and
                tracker.mean_appearance is not None):
                reid_score = tracker.get_appearance_similarity(detection.appearance_features)

            # Combined score: weight position and appearance
            if reid_score > self.doorway_config['reid_similarity_threshold']:
                final_score = position_dist * (1.0 - reid_score * 0.5)
            else:
                final_score = position_dist

            if final_score < best_score:
                best_score = final_score
                best_match_id = track_id

        return best_match_id

    def match_detection_to_tracker(self, detection: Detection) -> Optional[int]:
        """
        Match a detection to existing tracker using ByteTrack ID, ReID, and distance.
        Priority: ByteTrack ID > ReID similarity > Position distance
        """
        if not self.trackers:
            return None

        # Priority 1: If ByteTrack provides a track ID, try to match it first
        if detection.track_id is not None:
            # Check if we have a tracker with this ByteTrack ID
            if detection.track_id in self.trackers:
                return detection.track_id
            # ByteTrack assigned a new ID - this might be a new person or re-ID needed

        best_match_id = None
        best_score = float('inf')  # Lower is better for distance-based scoring

        # Priority 2 & 3: Combined ReID + distance matching
        for track_id, tracker in self.trackers.items():
            if len(tracker.positions) == 0:
                continue

            # Calculate distance to last known position
            last_pos = tracker.positions[-1]
            dist = math.sqrt(
                (detection.center[0] - last_pos[0])**2 +
                (detection.center[1] - last_pos[1])**2
            )

            # Consider velocity for prediction
            vx, vy = tracker.get_smooth_velocity()
            time_diff = detection.timestamp - tracker.last_update_time

            # Predict where tracker should be
            predicted_x = last_pos[0] + vx * time_diff
            predicted_y = last_pos[1] + vy * time_diff

            # Distance to predicted position
            pred_dist = math.sqrt(
                (detection.center[0] - predicted_x)**2 +
                (detection.center[1] - predicted_y)**2
            )

            # Use minimum of actual and predicted distance
            position_dist = min(dist, pred_dist)

            # Skip if too far away
            if position_dist > self.doorway_config['max_tracking_distance']:
                continue

            # Calculate ReID similarity score (if available)
            reid_score = 0.0
            if (self.doorway_config['use_reid'] and
                detection.appearance_features is not None and
                tracker.mean_appearance is not None):
                reid_score = tracker.get_appearance_similarity(detection.appearance_features)

            # Combined score: weight position and appearance
            # Lower score = better match
            # If ReID similarity is high (>0.5), reduce the effective distance
            if reid_score > self.doorway_config['reid_similarity_threshold']:
                # High appearance similarity - give significant boost
                final_score = position_dist * (1.0 - reid_score * 0.5)
            else:
                final_score = position_dist

            if final_score < best_score:
                best_score = final_score
                best_match_id = track_id

        return best_match_id

    def _find_best_reid_match(self, detection: Detection, max_distance: float = 200) -> Optional[int]:
        """
        Find the best ReID match for a detection among lost/stale trackers.
        Used when ByteTrack loses a track (e.g., during occlusion in groups).
        """
        if detection.appearance_features is None:
            return None

        best_match_id = None
        best_similarity = self.doorway_config['reid_similarity_threshold']

        for track_id, tracker in self.trackers.items():
            # Only consider trackers that haven't been updated recently (lost tracks)
            if tracker.time_since_update() < 0.5:
                continue  # Still being tracked, skip

            if tracker.mean_appearance is None:
                continue

            # Check distance isn't too far (sanity check)
            if len(tracker.positions) > 0:
                last_pos = tracker.positions[-1]
                dist = math.sqrt(
                    (detection.center[0] - last_pos[0])**2 +
                    (detection.center[1] - last_pos[1])**2
                )
                if dist > max_distance:
                    continue

            # Calculate appearance similarity
            similarity = tracker.get_appearance_similarity(detection.appearance_features)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = track_id

        return best_match_id

    def update_trackers(self, frame, detections: List[Detection]):
        """
        Update trackers with new detections and check for crossings.
        Uses ByteTrack IDs when available, falls back to ReID and position matching.

        Performance optimization: ReID features are only extracted for unmatched detections,
        not for every detection on every frame.
        """
        height, width = frame.shape[:2]
        current_time = time.time()

        # PHASE 1: Fast matching using ByteTrack IDs only (no ReID yet)
        matched_trackers = set()
        unmatched_detections = []

        for detection in detections:
            # Quick check: does ByteTrack ID match an existing tracker?
            if detection.track_id is not None and detection.track_id in self.trackers:
                track_id = detection.track_id
                # Update existing tracker
                self.trackers[track_id].update(detection)
                matched_trackers.add(track_id)

                # Check for crossing with trajectory validation
                self._check_and_handle_crossing(track_id, detection, width, current_time)
            else:
                # No ByteTrack match - will need ReID
                unmatched_detections.append(detection)

        # PHASE 2: Extract ReID features ONLY for unmatched detections (performance optimization)
        if unmatched_detections and self.doorway_config['use_reid'] and self.reid_extractor:
            # Collect bounding boxes for batch extraction
            reid_boxes = []
            for det in unmatched_detections:
                x, y, w, h = det.bbox
                reid_boxes.append((x, y, x + w, y + h))

            # Single batch forward pass for all unmatched detections
            features_list = self.reid_extractor.extract_batch(frame, reid_boxes)

            # Assign features to detections
            for i, det in enumerate(unmatched_detections):
                det.appearance_features = features_list[i]

        # PHASE 3: Try to match unmatched detections using ReID + position
        still_unmatched = []
        for detection in unmatched_detections:
            # Try position + ReID matching
            track_id = self._match_by_position_and_reid(detection)

            if track_id is not None:
                self.trackers[track_id].update(detection)
                matched_trackers.add(track_id)
                self._check_and_handle_crossing(track_id, detection, width, current_time)
            else:
                # Try ReID recovery for lost tracks
                if self.doorway_config['use_reid']:
                    reid_match = self._find_best_reid_match(detection)
                    if reid_match is not None:
                        self.trackers[reid_match].update(detection)
                        matched_trackers.add(reid_match)
                        self.logger.debug(f"ReID recovered track #{reid_match}")
                        continue
                still_unmatched.append(detection)

        # Create new trackers for truly unmatched detections
        for detection in still_unmatched:
            # Use ByteTrack ID if available, otherwise generate our own
            if detection.track_id is not None:
                new_id = detection.track_id
            elif self.track_id_pool:
                new_id = self.track_id_pool.pop(0)
            else:
                new_id = self.next_track_id
                self.next_track_id += 1

            # Create new tracker (ensure we don't overwrite existing)
            if new_id not in self.trackers:
                self.trackers[new_id] = PersonTracker(new_id, detection)

        # Check for crossings on unmatched trackers using interpolation
        # This catches crossings that may have occurred during dropped frames
        for track_id, tracker in self.trackers.items():
            if track_id in matched_trackers:
                continue  # Already processed

            time_gap = tracker.time_since_update()

            # Only interpolate if there's a significant time gap (dropped frames)
            # but not too stale (likely lost track)
            if 0.1 < time_gap < 1.0 and len(tracker.positions) >= 2:
                # Get interpolated positions through the gap
                interpolated = tracker.interpolate_positions(current_time, num_steps=5)

                for interp_pos in interpolated:
                    # Temporarily add interpolated position to check crossing
                    original_positions = list(tracker.positions)
                    tracker.positions.append(interp_pos)

                    crossing_dir = tracker.check_crossing(
                        width,
                        self.doorway_config['doorway_center'],
                        self.doorway_config['doorway_width'],
                        self.doorway_config['dual_line_gap']  # METHOD 3: pass dual_line_gap
                    )

                    # Restore original positions
                    tracker.positions = deque(original_positions, maxlen=60)

                    if crossing_dir and crossing_dir != CrossingDirection.NONE:
                        # Interpolation detected a crossing during frame gap
                        self.crossings[crossing_dir] += 1

                        # Make API call
                        if self.direction_flipped:
                            if crossing_dir == CrossingDirection.LEFT_TO_RIGHT:
                                threading.Thread(
                                    target=self.api_client.subtract_from_count,
                                    args=(1,),
                                    daemon=True
                                ).start()
                            elif crossing_dir == CrossingDirection.RIGHT_TO_LEFT:
                                threading.Thread(
                                    target=self.api_client.add_to_count,
                                    args=(1,),
                                    daemon=True
                                ).start()
                        else:
                            if crossing_dir == CrossingDirection.LEFT_TO_RIGHT:
                                threading.Thread(
                                    target=self.api_client.add_to_count,
                                    args=(1,),
                                    daemon=True
                                ).start()
                            elif crossing_dir == CrossingDirection.RIGHT_TO_LEFT:
                                threading.Thread(
                                    target=self.api_client.subtract_from_count,
                                    args=(1,),
                                    daemon=True
                                ).start()

                        total = sum(self.crossings.values())
                        self.logger.info(f"INTERPOLATED CROSSING #{total}: {crossing_dir.value} | "
                                       f"Track #{track_id} (gap: {time_gap:.2f}s) | "
                                       f"L->R: {self.crossings[CrossingDirection.LEFT_TO_RIGHT]} | "
                                       f"R->L: {self.crossings[CrossingDirection.RIGHT_TO_LEFT]}")
                        break  # Only count one crossing per interpolation check

        # Remove stale trackers
        trackers_to_remove = []
        for track_id, tracker in self.trackers.items():
            if tracker.time_since_update() > self.doorway_config['tracker_timeout']:
                trackers_to_remove.append(track_id)

                # Return ID to pool for reuse (only for non-ByteTrack IDs)
                if track_id not in self.track_id_pool and track_id < 1000:
                    self.track_id_pool.append(track_id)

        for track_id in trackers_to_remove:
            del self.trackers[track_id]

    def draw_visualization(self, frame, detections: List[Detection]) -> np.ndarray:
        """Draw visualization overlay on frame"""
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]

        # Calculate doorway line position
        center_x = int(width * self.doorway_config['doorway_center'])
        line_half_width = int(width * self.doorway_config['doorway_width'] / 2)

        # Draw approach zones
        approach_width = int(width * self.doorway_config['approach_zone'])

        # Left approach zone
        cv2.rectangle(vis_frame,
                     (center_x - approach_width - line_half_width, 0),
                     (center_x - line_half_width, height),
                     (100, 100, 200), 1)

        # Right approach zone
        cv2.rectangle(vis_frame,
                     (center_x + line_half_width, 0),
                     (center_x + approach_width + line_half_width, height),
                     (100, 100, 200), 1)

        # Draw doorway center line (bright green)
        cv2.line(vis_frame,
                (center_x, 0),
                (center_x, height),
                (0, 255, 0), 2)

        # Draw thin lines on either side to show doorway width
        cv2.line(vis_frame,
                (center_x - line_half_width, 0),
                (center_x - line_half_width, height),
                (0, 180, 0), 1)
        cv2.line(vis_frame,
                (center_x + line_half_width, 0),
                (center_x + line_half_width, height),
                (0, 180, 0), 1)

        # Draw current detections
        for detection in detections:
            x, y, w, h = detection.bbox

            # Color based on whether ByteTrack assigned an ID
            box_color = (0, 255, 255) if detection.track_id is not None else (255, 255, 0)

            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), box_color, 2)

            # Draw center point
            cv2.circle(vis_frame, detection.center, 5, (0, 255, 255), -1)

            # Draw confidence and ByteTrack ID if available
            if detection.track_id is not None:
                conf_text = f"BT#{detection.track_id} {detection.confidence:.2f}"
            else:
                conf_text = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(vis_frame, conf_text,
                       (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # Draw trackers
        for track_id, tracker in self.trackers.items():
            if len(tracker.positions) < 2:
                continue

            # Determine color based on side
            if tracker.current_side == 'left':
                track_color = (255, 100, 100)  # Blue-ish
            elif tracker.current_side == 'right':
                track_color = (100, 100, 255)  # Red-ish
            else:
                track_color = (0, 255, 0)  # Green

            # Draw path
            positions = list(tracker.positions)
            for i in range(1, len(positions)):
                pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                pt2 = (int(positions[i][0]), int(positions[i][1]))
                thickness = max(1, int(3 * (i / len(positions))))
                cv2.line(vis_frame, pt1, pt2, track_color, thickness)

            # Draw current position
            if positions:
                current_pos = positions[-1]
                cv2.circle(vis_frame,
                          (int(current_pos[0]), int(current_pos[1])),
                          8, track_color, -1)

                # Draw track ID, detection count, and crossing count
                info_text = f"#{track_id}"
                # Show detection count (important for trajectory validation)
                info_text += f" d:{tracker.detection_count}"
                if tracker.total_crossings > 0:
                    info_text += f" x{tracker.total_crossings}"

                cv2.putText(vis_frame, info_text,
                           (int(current_pos[0]) + 10, int(current_pos[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 2)

                # Draw velocity arrow
                vx, vy = tracker.get_smooth_velocity()
                if abs(vx) > 5 or abs(vy) > 5:
                    arrow_end = (int(current_pos[0] + vx * 0.5),
                               int(current_pos[1] + vy * 0.5))
                    cv2.arrowedLine(vis_frame,
                                  (int(current_pos[0]), int(current_pos[1])),
                                  arrow_end,
                                  (0, 255, 0), 2, tipLength=0.3)

        # Draw statistics overlay
        stats_bg_height = 160
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (0, 0), (380, stats_bg_height), (0, 0, 0), -1)
        vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)

        # Statistics text
        total_crossings = sum(self.crossings.values())
        lr_count = self.crossings[CrossingDirection.LEFT_TO_RIGHT]
        rl_count = self.crossings[CrossingDirection.RIGHT_TO_LEFT]
        net_flow = lr_count - rl_count

        # Mode indicator
        mode_text = "ByteTrack" if self.doorway_config['use_bytetrack'] else "Basic"
        if self.doorway_config['use_reid']:
            mode_text += "+ReID"

        cv2.putText(vis_frame, f"PEOPLE COUNTER [{mode_text}]",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Total Crossings: {total_crossings}",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Left -> Right: {lr_count}",
                   (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.putText(vis_frame, f"Right -> Left: {rl_count}",
                   (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        cv2.putText(vis_frame, f"Net Flow (L->R): {net_flow:+d}",
                   (10, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Active Tracks: {len(self.trackers)} | Detections: {len(detections)}",
                   (10, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return vis_frame

    def process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> Optional[np.ndarray]:
        """Process a single frame - main processing logic"""
        # Extract detections using YOLO
        detections = self.extract_detections(frame)

        # Update trackers and check for crossings
        self.update_trackers(frame, detections)

        # Create visualization
        vis_frame = self.draw_visualization(frame, detections)

        return vis_frame

