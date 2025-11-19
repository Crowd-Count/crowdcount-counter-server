"""
People Counter Processor
Enhanced overhead doorway person counter using YOLO for detection and tracking
"""

import cv2
import numpy as np
import time
import math
import threading
import requests
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from enum import Enum

from .base_processor import BaseStreamProcessor

# Try to import YOLO, provide helpful error if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


class CrossingDirection(Enum):
    """Direction of crossing the doorway line"""
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    NONE = "none"


@dataclass
class Detection:
    """Represents a single person detection"""
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    timestamp: float
    class_id: int
    class_name: str

    def __hash__(self):
        """Make Detection hashable for set operations"""
        return hash((self.center, self.timestamp))


class PersonTracker:
    """Enhanced person tracker with improved crossing detection"""

    def __init__(self, track_id: int, initial_detection: Detection):
        self.track_id = track_id
        self.detections = deque(maxlen=30)
        self.positions = deque(maxlen=30)
        self.velocities = deque(maxlen=10)
        self.creation_time = time.time()
        self.last_update_time = time.time()

        # Multiple crossing state tracking
        self.total_crossings = 0
        self.crossing_events = []  # List of (timestamp, direction) tuples
        self.last_side = None  # Track which side of center line person is on
        self.current_side = None
        self.last_crossing_time = 0
        self.min_crossing_interval = 0.3  # Minimum seconds between crossings

        # Overall tracking stats
        self.peak_confidence = 0.0

        # Add initial detection
        self.update(initial_detection)

    def update(self, detection: Detection):
        """Update tracker with new detection"""
        self.detections.append(detection)
        self.positions.append(detection.center)
        self.last_update_time = time.time()
        self.peak_confidence = max(self.peak_confidence, detection.confidence)

        # Calculate velocity
        if len(self.positions) >= 2:
            dx = self.positions[-1][0] - self.positions[-2][0]
            dy = self.positions[-1][1] - self.positions[-2][1]
            dt = detection.timestamp - self.detections[-2].timestamp if len(self.detections) >= 2 else 0.033
            if dt > 0:
                vx = dx / dt
                vy = dy / dt
                self.velocities.append((vx, vy))

    def check_crossing(self, frame_width: int, doorway_center: float, doorway_width: float) -> Optional[CrossingDirection]:
        """Check if a crossing has occurred and return the direction"""
        if len(self.positions) < 2:
            return None

        current_x = self.positions[-1][0]
        center_x = frame_width * doorway_center
        half_width = frame_width * doorway_width / 2

        # Determine which side of the center line the person is on
        if current_x < (center_x - half_width):
            self.current_side = 'left'
        elif current_x > (center_x + half_width):
            self.current_side = 'right'
        else:
            # Person is exactly on the line
            return None

        # Check if we have a previous side recorded
        if self.last_side is None:
            self.last_side = self.current_side
            return None

        # Check for crossing
        crossing_direction = None
        current_time = time.time()

        # Crossing from left to right
        if self.last_side == 'left' and self.current_side == 'right':
            if current_time - self.last_crossing_time >= self.min_crossing_interval:
                crossing_direction = CrossingDirection.LEFT_TO_RIGHT
                self.total_crossings += 1
                self.crossing_events.append((current_time, CrossingDirection.LEFT_TO_RIGHT))
                self.last_crossing_time = current_time

        # Crossing from right to left
        elif self.last_side == 'right' and self.current_side == 'left':
            if current_time - self.last_crossing_time >= self.min_crossing_interval:
                crossing_direction = CrossingDirection.RIGHT_TO_LEFT
                self.total_crossings += 1
                self.crossing_events.append((current_time, CrossingDirection.RIGHT_TO_LEFT))
                self.last_crossing_time = current_time

        # Update last side
        if self.current_side in ['left', 'right']:
            self.last_side = self.current_side

        return crossing_direction

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

    def __init__(self, stream_name: str, rtsp_url: str, device_info: Optional[dict] = None):
        super().__init__(stream_name, rtsp_url, device_info)

        # YOLO model setup
        self.model = None
        self.model_path = "yolo11n.pt"  # Default model, can be overridden
        self.class_names = {0: 'person'}  # Default

        # Doorway configuration
        self.doorway_config = {
            'doorway_center': 0.5,      # 50% of frame width (center line)
            'doorway_width': 0.02,      # 2% of frame width (narrow line)
            'approach_zone': 0.15,      # 15% approach zone on each side
            'min_path_length': 50,      # Minimum pixels traveled
            'max_tracking_distance': 120,  # Maximum distance for matching
            'tracker_timeout': 2.0,     # Seconds before removing stale tracker
            'yolo_conf_threshold': 0.3,  # YOLO confidence threshold
            'yolo_iou_threshold': 0.45   # YOLO IOU threshold for NMS
        }

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

        # API Configuration (optional)
        self.api_config = {
            'enabled': False,  # Disabled by default
            'base_url': 'https://api.crowdcount.tech',
            'user_id': 'Demo',
            'password': 'password',
            'timeout': 5
        }

    def on_start(self):
        """Initialize YOLO model when processor starts"""
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO not available - cannot start people counter")
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")

        try:
            self.logger.info(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)

            # Get class names from model
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                self.logger.info(f"Model classes: {', '.join([f'{k}:{v}' for k, v in self.class_names.items()])}")

            self.logger.info("YOLO model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

    def on_stop(self):
        """Cleanup when processor stops"""
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

    def extract_detections(self, frame) -> List[Detection]:
        """Extract person detections from frame using YOLO"""
        detections = []

        try:
            # Run YOLO inference
            results = self.model(
                frame,
                conf=self.doorway_config['yolo_conf_threshold'],
                iou=self.doorway_config['yolo_iou_threshold'],
                verbose=False
            )

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    # Process each detection
                    for box in result.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Calculate center and dimensions
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        width = int(x2 - x1)
                        height = int(y2 - y1)

                        # Get confidence and class
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.class_names.get(class_id, 'unknown')

                        # Create detection
                        detection = Detection(
                            center=(center_x, center_y),
                            bbox=(int(x1), int(y1), width, height),
                            confidence=confidence,
                            timestamp=time.time(),
                            class_id=class_id,
                            class_name=class_name
                        )

                        detections.append(detection)

        except Exception as e:
            self.logger.error(f"Detection error: {e}")

        return detections

    def match_detection_to_tracker(self, detection: Detection) -> Optional[int]:
        """Match a detection to existing tracker using distance metric"""
        if not self.trackers:
            return None

        best_match_id = None
        min_distance = self.doorway_config['max_tracking_distance']

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
            final_dist = min(dist, pred_dist)

            if final_dist < min_distance:
                min_distance = final_dist
                best_match_id = track_id

        return best_match_id

    def update_trackers(self, frame, detections: List[Detection]):
        """Update trackers with new detections and check for crossings"""
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

                # Check for crossing
                crossing_dir = self.trackers[track_id].check_crossing(
                    width,
                    self.doorway_config['doorway_center'],
                    self.doorway_config['doorway_width']
                )

                if crossing_dir and crossing_dir != CrossingDirection.NONE:
                    # A crossing occurred!
                    self.crossings[crossing_dir] += 1

                    # Make API call if enabled
                    if self.api_config['enabled']:
                        if crossing_dir == CrossingDirection.LEFT_TO_RIGHT:
                            threading.Thread(
                                target=self.api_add_to_count,
                                args=(1,),
                                daemon=True
                            ).start()
                        elif crossing_dir == CrossingDirection.RIGHT_TO_LEFT:
                            threading.Thread(
                                target=self.api_sub_from_count,
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
                        'position': detection.center
                    }
                    self.crossing_log.append(crossing_info)

                    # Log crossing event
                    total = sum(self.crossings.values())
                    self.logger.info(f"CROSSING #{total}: {crossing_dir.value} | "
                                   f"Track #{track_id} | "
                                   f"L→R: {self.crossings[CrossingDirection.LEFT_TO_RIGHT]} | "
                                   f"R→L: {self.crossings[CrossingDirection.RIGHT_TO_LEFT]}")
            else:
                unmatched_detections.append(detection)

        # Create new trackers for unmatched detections
        for detection in unmatched_detections:
            # Get new track ID
            if self.track_id_pool:
                new_id = self.track_id_pool.pop(0)
            else:
                new_id = self.next_track_id
                self.next_track_id += 1

            # Create new tracker
            self.trackers[new_id] = PersonTracker(new_id, detection)

        # Remove stale trackers
        trackers_to_remove = []
        for track_id, tracker in self.trackers.items():
            if tracker.time_since_update() > self.doorway_config['tracker_timeout']:
                trackers_to_remove.append(track_id)

                # Return ID to pool for reuse
                if track_id not in self.track_id_pool:
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

            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Draw center point
            cv2.circle(vis_frame, detection.center, 5, (0, 255, 255), -1)

            # Draw confidence
            conf_text = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(vis_frame, conf_text,
                       (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

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

                # Draw track ID and crossing count
                info_text = f"#{track_id}"
                if tracker.total_crossings > 0:
                    info_text += f" (x{tracker.total_crossings})"

                cv2.putText(vis_frame, info_text,
                           (int(current_pos[0]) + 10, int(current_pos[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 2)

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
        stats_bg_height = 140
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, stats_bg_height), (0, 0, 0), -1)
        vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)

        # Statistics text
        total_crossings = sum(self.crossings.values())
        lr_count = self.crossings[CrossingDirection.LEFT_TO_RIGHT]
        rl_count = self.crossings[CrossingDirection.RIGHT_TO_LEFT]
        net_flow = lr_count - rl_count

        cv2.putText(vis_frame, f"PEOPLE COUNTER - {self.stream_name}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Total Crossings: {total_crossings}",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Left -> Right: {lr_count}",
                   (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.putText(vis_frame, f"Right -> Left: {rl_count}",
                   (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        cv2.putText(vis_frame, f"Net Flow (L->R): {net_flow:+d}",
                   (10, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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

    def api_add_to_count(self, count: int = 1) -> bool:
        """Make API call to add to count (LEFT_TO_RIGHT crossing)"""
        if not self.api_config['enabled']:
            return True

        try:
            url = f"{self.api_config['base_url']}/add_to_count"
            data = {
                'user_id': self.api_config['user_id'],
                'password': self.api_config['password'],
                'count_to_add': str(count)
            }

            response = requests.post(url, data=data, timeout=self.api_config['timeout'])

            if response.status_code == 200:
                self.logger.info(f"API: Added {count} to remote count")
                return True
            else:
                self.logger.warning(f"API Error: {response.status_code}")
                return False

        except Exception as e:
            self.logger.warning(f"API Error: {e}")
            return False

    def api_sub_from_count(self, count: int = 1) -> bool:
        """Make API call to subtract from count (RIGHT_TO_LEFT crossing)"""
        if not self.api_config['enabled']:
            return True

        try:
            url = f"{self.api_config['base_url']}/sub_from_count"
            data = {
                'user_id': self.api_config['user_id'],
                'password': self.api_config['password'],
                'count_to_sub': str(count)
            }

            response = requests.post(url, data=data, timeout=self.api_config['timeout'])

            if response.status_code == 200:
                self.logger.info(f"API: Subtracted {count} from remote count")
                return True
            else:
                self.logger.warning(f"API Error: {response.status_code}")
                return False

        except Exception as e:
            self.logger.warning(f"API Error: {e}")
            return False
