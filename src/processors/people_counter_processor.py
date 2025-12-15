"""
People Counter Processor
Enhanced overhead doorway person counter using YOLO with ByteTrack tracking
Optimized for group detection with ReID and trajectory validation
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
    """Enhanced person tracker with trajectory validation, ReID, and improved crossing detection"""

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
        self.min_consecutive_for_crossing = 2  # Require N consecutive on new side

        # ReID appearance features
        self.appearance_history = deque(maxlen=10)  # Store recent appearance embeddings
        self.mean_appearance = None  # Running mean of appearance features

        # Pending crossing validation
        self.pending_crossing = None  # (direction, timestamp) if crossing needs validation
        self.crossing_start_position = None  # Position when crossing started

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
            # Update running mean of appearance
            if len(self.appearance_history) > 0:
                self.mean_appearance = np.mean(list(self.appearance_history), axis=0)

    def update_appearance(self, features: np.ndarray):
        """Update appearance features separately"""
        if features is not None:
            self.appearance_history.append(features)
            if len(self.appearance_history) > 0:
                self.mean_appearance = np.mean(list(self.appearance_history), axis=0)

    def get_appearance_similarity(self, features: np.ndarray) -> float:
        """Calculate cosine similarity with this tracker's appearance"""
        if self.mean_appearance is None or features is None:
            return 0.0
        # Cosine similarity
        dot_product = np.dot(self.mean_appearance, features)
        norm_a = np.linalg.norm(self.mean_appearance)
        norm_b = np.linalg.norm(features)
        if norm_a > 0 and norm_b > 0:
            return dot_product / (norm_a * norm_b)
        return 0.0

    def check_crossing(self, frame_width: int, doorway_center: float, doorway_width: float) -> Optional[CrossingDirection]:
        """
        Check if a crossing has occurred with trajectory validation.
        Requires minimum detections and consistent movement to count.
        """
        if len(self.positions) < 2:
            return None

        # Don't count crossings until we have enough detections (trajectory validation)
        if self.detection_count < self.min_detections_to_count:
            return None

        current_x = self.positions[-1][0]
        center_x = frame_width * doorway_center
        half_width = frame_width * doorway_width / 2

        # Determine which side of the center line the person is on
        previous_side = self.current_side
        if current_x < (center_x - half_width):
            self.current_side = 'left'
        elif current_x > (center_x + half_width):
            self.current_side = 'right'
        else:
            # Person is on the line - don't update side yet
            return None

        # Track side history for validation
        self.side_history.append(self.current_side)

        # Track consecutive detections on same side
        if previous_side == self.current_side:
            self.consecutive_side_detections += 1
        else:
            self.consecutive_side_detections = 1

        # Check if we have a previous side recorded
        if self.last_side is None:
            self.last_side = self.current_side
            return None

        # Check for crossing with validation
        crossing_direction = None
        current_time = time.time()

        # Detect potential crossing
        potential_crossing = None
        if self.last_side == 'left' and self.current_side == 'right':
            potential_crossing = CrossingDirection.LEFT_TO_RIGHT
        elif self.last_side == 'right' and self.current_side == 'left':
            potential_crossing = CrossingDirection.RIGHT_TO_LEFT

        if potential_crossing:
            # Validate crossing with trajectory analysis
            if self._validate_crossing(frame_width, doorway_center):
                if current_time - self.last_crossing_time >= self.min_crossing_interval:
                    crossing_direction = potential_crossing
                    self.total_crossings += 1
                    self.crossing_events.append((current_time, potential_crossing))
                    self.last_crossing_time = current_time

        # Update last side only after valid position
        if self.current_side in ['left', 'right']:
            self.last_side = self.current_side

        return crossing_direction

    def _validate_crossing(self, frame_width: int, doorway_center: float) -> bool:
        """
        Validate that a crossing is legitimate using trajectory analysis.
        Returns True if crossing should be counted.
        """
        # Require minimum consecutive detections on new side
        if self.consecutive_side_detections < self.min_consecutive_for_crossing:
            return False

        # Check for consistent movement direction in recent positions
        if len(self.positions) < 4:
            return True  # Not enough data, allow crossing

        recent_positions = list(self.positions)[-6:]  # Last 6 positions
        if len(recent_positions) < 3:
            return True

        # Calculate movement direction consistency
        x_movements = []
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            x_movements.append(dx)

        if not x_movements:
            return True

        # Check if movement is consistently in one direction (not bouncing)
        positive_moves = sum(1 for m in x_movements if m > 0)
        negative_moves = sum(1 for m in x_movements if m < 0)
        total_moves = positive_moves + negative_moves

        if total_moves == 0:
            return False  # No significant movement

        # Require at least 60% of movements in same direction
        consistency = max(positive_moves, negative_moves) / total_moves
        return consistency >= 0.6

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
            'yolo_conf_threshold': 0.25,  # TUNED: Lowered from 0.3 to catch more people in groups
            'yolo_iou_threshold': 0.3,   # TUNED: Lowered from 0.45 to separate overlapping people
            'min_crossing_interval': 0.5,  # TUNED: Increased from 0.3 to prevent double counts
            'reid_similarity_threshold': 0.5,  # NEW: Minimum similarity for ReID matching
            'use_bytetrack': True,       # NEW: Enable ByteTrack
            'use_reid': True,            # NEW: Enable ReID features
        }

        # ReID feature extractor (lazy loaded)
        self._reid_model = None

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
                    # Process each detection
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

                        # Extract ReID appearance features if enabled
                        appearance_features = None
                        if self.doorway_config['use_reid']:
                            appearance_features = self._extract_appearance_features(
                                frame, int(x1), int(y1), int(x2), int(y2)
                            )

                        # Create detection with track ID and ReID features
                        detection = Detection(
                            center=(center_x, center_y),
                            bbox=(int(x1), int(y1), width, height),
                            confidence=confidence,
                            timestamp=time.time(),
                            class_id=class_id,
                            class_name=class_name,
                            track_id=track_id,
                            appearance_features=appearance_features
                        )

                        detections.append(detection)

        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

        return detections

    def _extract_appearance_features(self, frame, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        """
        Extract appearance features from a person bounding box for ReID.
        Uses color histogram and simple CNN features for matching.
        """
        try:
            # Ensure valid coordinates
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            # Crop person region
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                return None

            # Resize to standard size for consistent features
            target_size = (64, 128)  # Width x Height (typical ReID aspect ratio)
            person_resized = cv2.resize(person_crop, target_size)

            # Extract multi-scale color histogram features (simple but effective)
            features = []

            # HSV color histogram (captures color regardless of lighting)
            hsv = cv2.cvtColor(person_resized, cv2.COLOR_BGR2HSV)

            # Split into upper and lower body (helps distinguish people)
            h_split = target_size[1] // 2
            upper = hsv[:h_split, :]
            lower = hsv[h_split:, :]

            for region in [upper, lower]:
                # Hue histogram (color)
                h_hist = cv2.calcHist([region], [0], None, [16], [0, 180])
                # Saturation histogram
                s_hist = cv2.calcHist([region], [1], None, [8], [0, 256])
                # Value histogram (brightness)
                v_hist = cv2.calcHist([region], [2], None, [8], [0, 256])

                # Normalize
                h_hist = h_hist.flatten() / (h_hist.sum() + 1e-6)
                s_hist = s_hist.flatten() / (s_hist.sum() + 1e-6)
                v_hist = v_hist.flatten() / (v_hist.sum() + 1e-6)

                features.extend([h_hist, s_hist, v_hist])

            # Combine all features
            feature_vector = np.concatenate(features)

            # L2 normalize the final feature vector
            norm = np.linalg.norm(feature_vector)
            if norm > 0:
                feature_vector = feature_vector / norm

            return feature_vector

        except Exception as e:
            self.logger.debug(f"ReID feature extraction error: {e}")
            return None

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
        """
        height, width = frame.shape[:2]
        current_time = time.time()

        # Match detections to existing trackers
        matched_trackers = set()
        unmatched_detections = []

        for detection in detections:
            track_id = self.match_detection_to_tracker(detection)

            if track_id is not None:
                # Update existing tracker
                self.trackers[track_id].update(detection)
                matched_trackers.add(track_id)

                # Check for crossing with trajectory validation
                crossing_dir = self.trackers[track_id].check_crossing(
                    width,
                    self.doorway_config['doorway_center'],
                    self.doorway_config['doorway_width']
                )

                if crossing_dir and crossing_dir != CrossingDirection.NONE:
                    # A validated crossing occurred!
                    self.crossings[crossing_dir] += 1

                    # Make API call with device credentials
                    # Apply direction flip logic if enabled
                    if self.direction_flipped:
                        # Direction is flipped: LEFT->RIGHT becomes exit, RIGHT->LEFT becomes entry
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
                        # Normal direction: LEFT->RIGHT is entry, RIGHT->LEFT is exit
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

                    # Log crossing event with trajectory info
                    total = sum(self.crossings.values())
                    self.logger.info(f"CROSSING #{total}: {crossing_dir.value} | "
                                   f"Track #{track_id} (detections: {self.trackers[track_id].detection_count}) | "
                                   f"L->R: {self.crossings[CrossingDirection.LEFT_TO_RIGHT]} | "
                                   f"R->L: {self.crossings[CrossingDirection.RIGHT_TO_LEFT]}")
            else:
                unmatched_detections.append(detection)

        # Try to recover unmatched detections using ReID (for group occlusions)
        still_unmatched = []
        for detection in unmatched_detections:
            if self.doorway_config['use_reid']:
                # Try to find a lost tracker using appearance matching
                reid_match = self._find_best_reid_match(detection)
                if reid_match is not None:
                    # Recovered a lost track via ReID
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

