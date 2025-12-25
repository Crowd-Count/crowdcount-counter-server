"""
People Counter Processor - Method 2: Zone-Based State Machine
Enhanced overhead doorway person counter using YOLO with ByteTrack tracking

METHOD 2: Zone-Based State Machine with Dwell Time
- Three zones: OUTSIDE_LEFT, THRESHOLD, OUTSIDE_RIGHT
- State machine: IDLE → APPROACHING → IN_THRESHOLD → CROSSED → CONFIRMED
- Dwell time requirements in each zone prevent noise/jitter
- State resets on timeout or return to origin zone
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


class CrossingState(Enum):
    """METHOD 2: State machine states for zone-based crossing detection"""
    IDLE = "idle"                   # Not in crossing sequence
    APPROACHING = "approaching"      # Moving toward threshold from origin zone
    IN_THRESHOLD = "in_threshold"    # Currently in threshold zone
    CROSSED = "crossed"              # Exited threshold to destination zone
    CONFIRMED = "confirmed"          # Dwell time in destination met, ready to emit


class Zone(Enum):
    """METHOD 2: Zone definitions"""
    OUTSIDE_LEFT = "outside_left"
    THRESHOLD = "threshold"
    OUTSIDE_RIGHT = "outside_right"


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
    METHOD 2: Zone-Based State Machine with Dwell Time

    Key features:
    - Three zones: OUTSIDE_LEFT, THRESHOLD, OUTSIDE_RIGHT
    - State machine tracks crossing progress: IDLE → APPROACHING → IN_THRESHOLD → CROSSED → CONFIRMED
    - Dwell time requirements in each zone filter noise
    - State resets on timeout or return to origin
    """

    # Method 2 configuration constants
    DWELL_TIME_ORIGIN = 0.2       # Minimum seconds in origin zone before APPROACHING
    DWELL_TIME_THRESHOLD = 0.1    # Minimum seconds in threshold zone
    DWELL_TIME_DESTINATION = 0.3  # Minimum seconds in destination zone to CONFIRM
    STATE_TIMEOUT = 5.0           # Maximum seconds in any state before reset
    TRACK_LOST_TIMEOUT = 1.0      # Maximum seconds track can be lost before reset

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

        # METHOD 2: State machine tracking
        self.crossing_state = CrossingState.IDLE
        self.current_zone = None                    # Current zone (Zone enum)
        self.origin_zone = None                     # Zone where crossing started
        self.expected_destination = None            # Expected destination zone
        self.zone_entry_time = None                 # When we entered current zone
        self.state_entry_time = None                # When we entered current state
        self.crossing_direction = None              # Direction of in-progress crossing

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

    def _determine_zone(self, x: int, frame_width: int, doorway_center: float,
                        doorway_width: float, approach_zone: float) -> Zone:
        """METHOD 2: Determine which zone the x-coordinate is in."""
        center_x = frame_width * doorway_center
        threshold_half = frame_width * doorway_width / 2

        if x < center_x - threshold_half:
            return Zone.OUTSIDE_LEFT
        elif x > center_x + threshold_half:
            return Zone.OUTSIDE_RIGHT
        else:
            return Zone.THRESHOLD

    def _reset_state_machine(self):
        """METHOD 2: Reset the state machine to IDLE."""
        self.crossing_state = CrossingState.IDLE
        self.origin_zone = None
        self.expected_destination = None
        self.state_entry_time = None
        self.crossing_direction = None

    def check_crossing(self, frame_width: int, doorway_center: float, doorway_width: float,
                       approach_zone: float = 0.15) -> Optional[CrossingDirection]:
        """
        METHOD 2: Zone-Based State Machine with Dwell Time

        State machine transitions:
        IDLE → APPROACHING: When in origin zone with sufficient dwell time
        APPROACHING → IN_THRESHOLD: When entering threshold zone
        IN_THRESHOLD → CROSSED: When exiting threshold to destination zone
        CROSSED → CONFIRMED: When dwell time in destination met

        Reset conditions:
        - Return to origin zone before completing sequence
        - Timeout in any state > 5 seconds
        - Track lost for > 1 second
        """
        if len(self.positions) < 2:
            return None

        current_x = self.positions[-1][0]
        current_time = time.time()

        # Determine current zone
        new_zone = self._determine_zone(current_x, frame_width, doorway_center,
                                        doorway_width, approach_zone)

        # Check for zone change
        zone_changed = (new_zone != self.current_zone)
        if zone_changed:
            self.zone_entry_time = current_time

        previous_zone = self.current_zone
        self.current_zone = new_zone

        # Update legacy side tracking for compatibility
        if new_zone == Zone.OUTSIDE_LEFT:
            self.current_side = 'left'
        elif new_zone == Zone.OUTSIDE_RIGHT:
            self.current_side = 'right'

        # Check for state timeout
        if self.state_entry_time and (current_time - self.state_entry_time > self.STATE_TIMEOUT):
            self._reset_state_machine()

        # METHOD 2: State machine logic
        crossing_to_emit = None

        if self.crossing_state == CrossingState.IDLE:
            # In IDLE, wait for person to be in an outside zone with dwell time
            if new_zone in (Zone.OUTSIDE_LEFT, Zone.OUTSIDE_RIGHT):
                dwell_time = current_time - self.zone_entry_time if self.zone_entry_time else 0

                if dwell_time >= self.DWELL_TIME_ORIGIN:
                    # Start approaching - determine expected crossing direction
                    self.crossing_state = CrossingState.APPROACHING
                    self.state_entry_time = current_time
                    self.origin_zone = new_zone

                    if new_zone == Zone.OUTSIDE_LEFT:
                        self.expected_destination = Zone.OUTSIDE_RIGHT
                        self.crossing_direction = CrossingDirection.LEFT_TO_RIGHT
                    else:
                        self.expected_destination = Zone.OUTSIDE_RIGHT
                        self.crossing_direction = CrossingDirection.RIGHT_TO_LEFT
                        self.expected_destination = Zone.OUTSIDE_LEFT

        elif self.crossing_state == CrossingState.APPROACHING:
            # In APPROACHING, wait for entry into threshold zone
            if new_zone == Zone.THRESHOLD:
                self.crossing_state = CrossingState.IN_THRESHOLD
                self.state_entry_time = current_time
            elif new_zone == self.origin_zone:
                # Still in origin zone, that's okay - continue waiting
                pass
            elif new_zone == self.expected_destination:
                # Skipped threshold somehow (fast movement or detection gap)
                # Go directly to CROSSED
                self.crossing_state = CrossingState.CROSSED
                self.state_entry_time = current_time

        elif self.crossing_state == CrossingState.IN_THRESHOLD:
            # In threshold zone, wait for dwell time then exit
            dwell_time = current_time - self.zone_entry_time if self.zone_entry_time else 0

            if new_zone == self.expected_destination:
                if dwell_time >= self.DWELL_TIME_THRESHOLD:
                    # Exited threshold to destination with sufficient dwell
                    self.crossing_state = CrossingState.CROSSED
                    self.state_entry_time = current_time
                else:
                    # Exited too quickly - still count but note it
                    self.crossing_state = CrossingState.CROSSED
                    self.state_entry_time = current_time
            elif new_zone == self.origin_zone:
                # Returned to origin - reset (U-turn)
                self._reset_state_machine()

        elif self.crossing_state == CrossingState.CROSSED:
            # Exited threshold, now waiting for dwell time in destination
            if new_zone == self.expected_destination:
                dwell_time = current_time - self.zone_entry_time if self.zone_entry_time else 0

                if dwell_time >= self.DWELL_TIME_DESTINATION:
                    # Confirmed! Emit crossing
                    self.crossing_state = CrossingState.CONFIRMED
                    self.state_entry_time = current_time

                    # Check minimum interval between crossings
                    if current_time - self.last_crossing_time >= self.min_crossing_interval:
                        crossing_to_emit = self.crossing_direction
                        self.total_crossings += 1
                        self.crossing_events.append((current_time, self.crossing_direction))
                        self.last_crossing_time = current_time
                        self.last_side = self.current_side

                    # Reset for next crossing
                    self._reset_state_machine()

            elif new_zone == Zone.THRESHOLD:
                # Back in threshold - could be oscillating, wait
                pass
            elif new_zone == self.origin_zone:
                # Returned all the way to origin - reset
                self._reset_state_machine()

        elif self.crossing_state == CrossingState.CONFIRMED:
            # Should have been reset already, but just in case
            self._reset_state_machine()

        return crossing_to_emit

    def _validate_crossing(self, frame_width: int, doorway_center: float) -> bool:
        """Legacy validation method - always returns True for Method 2."""
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
            # METHOD 2: Zone-Based State Machine parameters
            'dwell_time_origin': 0.2,       # Min seconds in origin zone
            'dwell_time_threshold': 0.1,    # Min seconds in threshold zone
            'dwell_time_destination': 0.3,  # Min seconds in destination to confirm
            'state_timeout': 5.0,           # Max seconds in any state before reset
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
            self.doorway_config['approach_zone']  # METHOD 2: pass approach_zone for zone determination
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
                        self.doorway_config['approach_zone']  # METHOD 2: pass approach_zone
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

