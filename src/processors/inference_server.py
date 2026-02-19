"""
Centralized GPU Inference Server

One instance per GPU. Handles YOLO detection + ReID feature extraction
for all cameras assigned to it, eliminating per-process CUDA context overhead.

Architecture:
    Camera Workers (CPU only) send frames via multiprocessing.Queue to the
    inference server, which runs YOLO predict + ReID extract_batch on GPU
    and returns results via per-camera result queues.
"""

import time
import logging
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
from pathlib import Path
from uuid import uuid4
import numpy as np

# Get project root directory (crowdcount-counter-server/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class RequestType(Enum):
    """Types of inference requests"""
    DETECT = "detect"
    REID = "reid"
    REGISTER = "register"    # Register camera_id -> slot mapping
    UNREGISTER = "unregister"  # Unregister camera
    SHUTDOWN = "shutdown"


@dataclass
class InferenceRequest:
    """Request sent from camera worker to inference server.

    Note: result queues are NOT sent in requests because multiprocessing.Queue
    cannot be pickled through another Queue. Instead, result queues are
    pre-allocated as a pool and inherited by the child process at spawn time.
    Each camera is assigned a slot, and REGISTER requests map camera_id to slot.
    """
    type: RequestType
    camera_id: str
    frame: Optional[np.ndarray] = None
    # For DETECT requests
    conf: float = 0.25
    iou: float = 0.3
    imgsz: int = 480
    # For REID requests
    boxes: Optional[List[Tuple[int, int, int, int]]] = None
    # For REGISTER requests
    slot_name: Optional[str] = None
    # Timestamp for staleness checking
    created_at: float = field(default_factory=time.time)
    # Unique ID for request-response correlation
    request_id: str = field(default_factory=lambda: uuid4().hex[:12])


@dataclass
class DetectionResult:
    """Single detection returned from inference server"""
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # x1, y1, width, height
    confidence: float
    class_id: int
    class_name: str


@dataclass
class InferenceResponse:
    """Response sent from inference server back to camera worker"""
    type: RequestType
    camera_id: str
    # For DETECT responses
    detections: Optional[List[DetectionResult]] = None
    # For REID responses
    features: Optional[List[Optional[np.ndarray]]] = None
    # Error info
    error: Optional[str] = None
    # Correlation ID from the originating request
    request_id: Optional[str] = None


class InferenceServer:
    """
    Centralized GPU inference process.
    One instance per GPU. Handles YOLO detection + ReID for all cameras assigned to it.
    """

    def __init__(
        self,
        gpu_id: int,
        model_path: str,
        reid_model_name: str,
        request_queue: multiprocessing.Queue,
        result_queues: Dict[str, multiprocessing.Queue],
        config: dict,
        heartbeat: Optional[multiprocessing.Value] = None,
    ):
        """
        Initialize inference server configuration.

        Args:
            gpu_id: CUDA device index (0, 1, ...)
            model_path: Path to YOLO model weights
            reid_model_name: Name of ReID model (e.g., 'osnet_x0_25')
            request_queue: Shared input queue for all cameras
            result_queues: Dict mapping camera slot name to pre-allocated result queue.
                          Created before spawn and inherited by the child process.
                          Keys are slot names like "slot_0", "slot_1", etc.
            config: Configuration dict with inference parameters
            heartbeat: Shared multiprocessing.Value('d') for liveness monitoring.
                      Updated after each main loop iteration so the orchestrator can
                      detect stuck inference (CUDA hang, OOM) even when is_alive() is True.
        """
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.reid_model_name = reid_model_name
        self.request_queue = request_queue
        self.result_queues = result_queues  # Pre-allocated pool, inherited at spawn
        self.config = config
        self.heartbeat = heartbeat
        self.running = False
        # Maps camera_id to slot name, populated dynamically as cameras register
        self.camera_to_slot: Dict[str, str] = {}

        # These are initialized in run() after CUDA context is set
        self.model = None
        self.reid_extractor = None
        self.class_names = {0: 'person'}

        # Stats
        self.requests_processed = 0
        self.start_time = None

        # Logger
        self.logger = logging.getLogger(f"InferenceServer-GPU{gpu_id}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - InferenceServer-GPU{gpu_id} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def run(self):
        """
        Main inference loop. Call this as the target of multiprocessing.Process.
        Initializes CUDA context, loads models, and processes requests.
        """
        import torch
        from ultralytics import YOLO
        from .deep_reid import create_reid_extractor

        self.running = True
        self.start_time = time.time()

        # Set CUDA device for this process
        device_str = f'cuda:{self.gpu_id}'
        torch.cuda.set_device(self.gpu_id)
        self.logger.info(f"Initializing on {device_str} ({torch.cuda.get_device_name(self.gpu_id)})")

        # Load YOLO model
        self.logger.info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.model.to(device_str)
        model_device = next(self.model.model.parameters()).device
        self.logger.info(f"YOLO model loaded on device: {model_device}")

        # Get class names
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names

        # Load ReID model
        if self.reid_model_name:
            self.logger.info(f"Loading ReID model: {self.reid_model_name}")
            self.reid_extractor = create_reid_extractor(
                model_name=self.reid_model_name,
                device=device_str,
                verbose=True
            )
            self.logger.info(f"ReID model loaded (feature dim: {self.reid_extractor.feature_dim})")

        self.logger.info("Inference server ready, waiting for requests...")

        # Main processing loop
        last_stats_time = time.time()
        while self.running:
            try:
                # Block waiting for request with timeout so we can check self.running
                try:
                    request = self.request_queue.get(timeout=1.0)
                except Exception:
                    # Queue.get timeout - update heartbeat to prove loop is alive even when idle
                    if self.heartbeat is not None:
                        self.heartbeat.value = time.time()
                    continue

                if request.type == RequestType.SHUTDOWN:
                    self.logger.info("Received shutdown request")
                    self.running = False
                    break

                if request.type == RequestType.REGISTER:
                    # Map camera_id to its pre-allocated result queue slot
                    self.camera_to_slot[request.camera_id] = request.slot_name
                    self.logger.info(f"Registered camera {request.camera_id} -> slot {request.slot_name}")
                    continue

                if request.type == RequestType.UNREGISTER:
                    self.camera_to_slot.pop(request.camera_id, None)
                    self.logger.info(f"Unregistered camera {request.camera_id}")
                    continue

                # Look up result queue for this camera via slot mapping
                slot_name = self.camera_to_slot.get(request.camera_id)
                if slot_name is None:
                    self.logger.warning(f"Camera {request.camera_id} not registered, dropping request")
                    continue
                result_queue = self.result_queues.get(slot_name)
                if result_queue is None:
                    self.logger.warning(f"No result queue for slot {slot_name}, dropping request")
                    continue

                # Skip stale requests — drop before GPU inference so the
                # response doesn't arrive after the processor's 5s timeout.
                # Using 3s threshold leaves ~2s headroom for inference.
                if request.type in (RequestType.DETECT, RequestType.REID):
                    age = time.time() - request.created_at
                    if age > 3.0:
                        continue

                if request.type == RequestType.DETECT:
                    response = self._handle_detect(request)
                elif request.type == RequestType.REID:
                    response = self._handle_reid(request)
                else:
                    response = InferenceResponse(
                        type=request.type,
                        camera_id=request.camera_id,
                        error=f"Unknown request type: {request.type}",
                        request_id=request.request_id,
                    )

                # Send response back to camera worker
                try:
                    result_queue.put_nowait(response)
                except Exception as e:
                    self.logger.error(f"Failed to deliver result for camera {request.camera_id}: {e}")

                self.requests_processed += 1

                # Update heartbeat after each processed request
                if self.heartbeat is not None:
                    self.heartbeat.value = time.time()

                # Log stats periodically
                now = time.time()
                if now - last_stats_time >= 10.0:
                    elapsed = now - self.start_time
                    rps = self.requests_processed / elapsed if elapsed > 0 else 0
                    queue_size = self.request_queue.qsize() if hasattr(self.request_queue, 'qsize') else -1
                    self.logger.info(
                        f"Stats: {self.requests_processed} requests | "
                        f"{rps:.1f} req/s | queue depth: {queue_size}"
                    )
                    last_stats_time = now

            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

        # Cleanup
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info(
            f"Inference server stopped. Processed {self.requests_processed} "
            f"requests in {elapsed:.1f}s"
        )

    def _handle_detect(self, request: InferenceRequest) -> InferenceResponse:
        """Handle a YOLO detection request"""
        try:
            results = self.model.predict(
                request.frame,
                conf=request.conf,
                iou=request.iou,
                classes=[0],  # Only detect person class
                half=True,
                imgsz=request.imgsz,
                verbose=False
            )

            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.class_names.get(class_id, 'unknown')

                        if class_id != 0:
                            continue

                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        width = int(x2 - x1)
                        height = int(y2 - y1)

                        detections.append(DetectionResult(
                            center=(center_x, center_y),
                            bbox=(int(x1), int(y1), width, height),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name,
                        ))

            return InferenceResponse(
                type=RequestType.DETECT,
                camera_id=request.camera_id,
                detections=detections,
                request_id=request.request_id,
            )

        except Exception as e:
            self.logger.error(f"Detection error for {request.camera_id}: {e}")
            return InferenceResponse(
                type=RequestType.DETECT,
                camera_id=request.camera_id,
                detections=[],
                error=str(e),
                request_id=request.request_id,
            )

    def _handle_reid(self, request: InferenceRequest) -> InferenceResponse:
        """Handle a ReID feature extraction request"""
        try:
            if self.reid_extractor is None:
                return InferenceResponse(
                    type=RequestType.REID,
                    camera_id=request.camera_id,
                    features=[None] * len(request.boxes) if request.boxes else [],
                    error="ReID extractor not available",
                    request_id=request.request_id,
                )

            features = self.reid_extractor.extract_batch(request.frame, request.boxes or [])

            return InferenceResponse(
                type=RequestType.REID,
                camera_id=request.camera_id,
                features=features,
                request_id=request.request_id,
            )

        except Exception as e:
            self.logger.error(f"ReID error for {request.camera_id}: {e}")
            return InferenceResponse(
                type=RequestType.REID,
                camera_id=request.camera_id,
                features=[None] * len(request.boxes) if request.boxes else [],
                error=str(e),
                request_id=request.request_id,
            )

