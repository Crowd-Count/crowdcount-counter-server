"""
Centralized GPU Inference Server

One instance per GPU. Handles YOLO detection + ReID feature extraction
for all cameras assigned to it, eliminating per-process CUDA context overhead.

Architecture:
    Camera Workers (CPU only) send frames via multiprocessing.Queue to the
    inference server, which runs YOLO predict + ReID extract_batch on GPU
    and returns results via per-camera result queues.

    Frames are collected into batches (up to INFERENCE_BATCH_SIZE) and
    processed in a single GPU call for ~3x throughput improvement.
"""

import time
import logging
import multiprocessing
import shutil
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from enum import Enum
from pathlib import Path
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


class InferenceServer:
    """
    Centralized GPU inference process.
    One instance per GPU. Handles YOLO detection + ReID for all cameras assigned to it.
    Uses batched inference for ~3x throughput over serial processing.
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

        # Batch settings
        self.batch_size = config.get('batch_size', 8)
        self.batch_wait_ms = config.get('batch_wait_ms', 30)

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

    # -------------------------------------------------------------------------
    # TensorRT engine management
    # -------------------------------------------------------------------------

    def _get_engine_dir(self) -> Path:
        """Determine the directory for storing TensorRT engines.

        Priority: explicit config dir > models/ dir (if writable) > data/engines/ fallback.
        """
        explicit_dir = self.config.get('tensorrt_engine_dir', '')
        if explicit_dir:
            d = Path(explicit_dir)
            d.mkdir(parents=True, exist_ok=True)
            return d

        # Try models/ directory first
        if MODELS_DIR.exists():
            try:
                test_file = MODELS_DIR / ".write_test"
                test_file.touch()
                test_file.unlink()
                return MODELS_DIR
            except OSError:
                pass

        # Fallback to data/engines/
        fallback = PROJECT_ROOT / "data" / "engines"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _resolve_model_path(self, device_str: str) -> str:
        """Resolve the model path, preferring a TensorRT engine if available.

        Checks if TensorRT is enabled, builds a GPU-specific engine filename,
        and either returns the existing engine or triggers auto-export.
        Falls back to the original .pt path on any failure.
        """
        import torch

        if not self.config.get('tensorrt_enabled', True):
            self.logger.info("TensorRT disabled by config, using PyTorch model")
            return self.model_path

        if not self.model_path.endswith('.pt'):
            self.logger.info(f"Model is not a .pt file ({self.model_path}), skipping TensorRT")
            return self.model_path

        try:
            # Get GPU compute capability
            cap = torch.cuda.get_device_capability(self.gpu_id)
            gpu_arch = f"sm{cap[0]}{cap[1]}"

            max_batch = self.config.get('tensorrt_max_batch', self.batch_size)
            imgsz = self.config.get('imgsz', 480)

            # Build GPU-specific engine filename
            model_stem = Path(self.model_path).stem
            engine_name = f"{model_stem}_{gpu_arch}_b{max_batch}_fp16_imgsz{imgsz}.engine"

            engine_dir = self._get_engine_dir()
            engine_path = engine_dir / engine_name

            if engine_path.exists():
                self.logger.info(f"Found existing TensorRT engine: {engine_path}")
                return str(engine_path)

            self.logger.info(
                f"No TensorRT engine found for {gpu_arch}, exporting: {engine_path}"
            )
            return self._export_tensorrt_engine(engine_path, device_str, max_batch, imgsz)

        except Exception as e:
            self.logger.error(f"TensorRT resolution failed, falling back to PyTorch: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self.model_path

    def _export_tensorrt_engine(
        self, engine_path: Path, device_str: str, max_batch: int, imgsz: int
    ) -> str:
        """Export a TensorRT engine from the .pt weights.

        Copies .pt to the writable engine dir (avoids read-only /app mount),
        calls ultralytics export, renames the result to the GPU-specific name,
        and cleans up temp files.

        Returns the engine path on success, or falls back to .pt on failure.
        """
        from ultralytics import YOLO

        try:
            engine_dir = engine_path.parent
            pt_name = Path(self.model_path).name

            # Copy .pt to engine dir so ultralytics writes the .engine there
            temp_pt = engine_dir / pt_name
            if not temp_pt.exists() or str(temp_pt) != self.model_path:
                shutil.copy2(self.model_path, temp_pt)

            self.logger.info(
                f"Exporting TensorRT engine (batch={max_batch}, imgsz={imgsz}, fp16=True)... "
                f"This may take several minutes on first run."
            )

            export_model = YOLO(str(temp_pt))
            exported_path = export_model.export(
                format="engine",
                dynamic=True,
                half=True,
                imgsz=imgsz,
                batch=max_batch,
                device=device_str,
            )

            # ultralytics returns the path to the exported engine
            exported = Path(exported_path)
            if not exported.exists():
                raise FileNotFoundError(f"Export succeeded but file not found: {exported}")

            # Rename to our GPU-specific name
            if exported != engine_path:
                shutil.move(str(exported), str(engine_path))

            # Clean up temp .pt copy (only if we copied it)
            if str(temp_pt) != self.model_path and temp_pt.exists():
                temp_pt.unlink()

            # Clean up any intermediate ONNX files
            onnx_file = engine_dir / Path(self.model_path).with_suffix('.onnx').name
            if onnx_file.exists():
                onnx_file.unlink()
                self.logger.info(f"Cleaned up intermediate ONNX: {onnx_file}")

            self.logger.info(f"TensorRT engine exported successfully: {engine_path}")
            return str(engine_path)

        except Exception as e:
            self.logger.error(f"TensorRT export failed, falling back to PyTorch: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self.model_path

    def _warmup_model(self, device_str: str):
        """Run dummy inference passes to initialize TensorRT CUDA kernels.

        Prevents latency spike on first real inference.
        """
        warmup_iters = self.config.get('tensorrt_warmup_iters', 3)
        if warmup_iters <= 0:
            return

        imgsz = self.config.get('imgsz', 480)
        self.logger.info(f"Warming up model with {warmup_iters} dummy inference passes...")

        try:
            dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
            for i in range(warmup_iters):
                self.model.predict(
                    dummy_frame,
                    conf=0.25,
                    iou=0.3,
                    classes=[0],
                    half=True,
                    imgsz=imgsz,
                    verbose=False,
                )
            self.logger.info("Model warmup complete")
        except Exception as e:
            self.logger.warning(f"Warmup failed (non-fatal): {e}")

    # -------------------------------------------------------------------------
    # Main inference loop
    # -------------------------------------------------------------------------

    def run(self):
        """
        Main inference loop. Call this as the target of multiprocessing.Process.
        Initializes CUDA context, loads models, and processes requests in batches.
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

        # Resolve model path (TensorRT engine or PyTorch .pt)
        resolved_path = self._resolve_model_path(device_str)
        is_engine = resolved_path.endswith('.engine')

        # Load YOLO model
        self.logger.info(f"Loading YOLO model: {resolved_path}")
        try:
            if is_engine:
                self.model = YOLO(resolved_path, task='detect')
            else:
                self.model = YOLO(resolved_path)
                self.model.to(device_str)
        except Exception as e:
            if is_engine:
                self.logger.error(f"TensorRT engine load failed: {e}, falling back to PyTorch")
                self.model = YOLO(self.model_path)
                self.model.to(device_str)
                is_engine = False
            else:
                raise

        backend = "TensorRT" if is_engine else "PyTorch"
        if is_engine:
            self.logger.info(f"YOLO model loaded on {device_str} (backend: {backend})")
        else:
            model_device = next(self.model.model.parameters()).device
            self.logger.info(f"YOLO model loaded on device: {model_device} (backend: {backend})")

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

        # Warmup model (initializes TensorRT CUDA kernels, prevents first-inference spike)
        self._warmup_model(device_str)

        self.logger.info(
            f"Inference server ready (batch_size={self.batch_size}, "
            f"batch_wait_ms={self.batch_wait_ms}, backend={backend}), waiting for requests..."
        )

        # Main processing loop
        last_stats_time = time.time()
        last_registration_log_time = time.time()
        while self.running:
            try:
                batch = self._collect_batch()
                if not batch:
                    continue

                # Separate DETECT vs REID requests
                detect_batch = [(r, s) for r, s in batch if r.type == RequestType.DETECT]
                reid_batch = [(r, s) for r, s in batch if r.type == RequestType.REID]

                if detect_batch:
                    self._process_detect_batch(detect_batch)
                if reid_batch:
                    self._process_reid_batch(reid_batch)

                self.requests_processed += len(batch)

                # Update heartbeat after each batch
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
                        f"{rps:.1f} req/s | queue depth: {queue_size} | "
                        f"last batch: {len(batch)}"
                    )
                    last_stats_time = now

                # Periodic registration summary (every 60s)
                if now - last_registration_log_time >= 60.0:
                    registered = list(self.camera_to_slot.keys())
                    self.logger.info(f"Registered cameras ({len(registered)}): {registered}")
                    last_registration_log_time = now

            except Exception as e:
                self.logger.error(f"Error in batch processing: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

        # Cleanup
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info(
            f"Inference server stopped. Processed {self.requests_processed} "
            f"requests in {elapsed:.1f}s"
        )

    # -------------------------------------------------------------------------
    # Batch collection
    # -------------------------------------------------------------------------

    def _collect_batch(self):
        """Collect a batch of DETECT/REID requests from the queue.

        Returns as soon as batch_size reached or batch_wait_ms elapsed.
        Handles REGISTER/UNREGISTER/SHUTDOWN inline (not batched).
        Returns list of (request, slot_name) tuples to process.
        """
        batch = []
        deadline = time.time() + self.batch_wait_ms / 1000.0

        while len(batch) < self.batch_size:
            remaining = deadline - time.time()
            if remaining <= 0 and batch:
                break  # Time's up, process what we have

            try:
                timeout = max(0.001, remaining) if batch else 1.0
                request = self.request_queue.get(timeout=timeout)
            except Exception:
                if batch:
                    break  # Queue empty, process what we have
                # Update heartbeat while idle
                if self.heartbeat is not None:
                    self.heartbeat.value = time.time()
                continue

            # Handle control messages immediately (not batched)
            if request.type == RequestType.SHUTDOWN:
                self.running = False
                return []

            if request.type == RequestType.REGISTER:
                self._handle_register(request)
                continue

            if request.type == RequestType.UNREGISTER:
                self._handle_unregister(request)
                continue

            # Validate camera is registered
            slot_name = self.camera_to_slot.get(request.camera_id)
            if slot_name is None:
                self._send_unregistered_error(request)
                continue

            batch.append((request, slot_name))
            if len(batch) == 1:
                # Start the batch window from the first real request, not from
                # function entry (where the idle queue.get may have burned it).
                deadline = time.time() + self.batch_wait_ms / 1000.0

        return batch

    # -------------------------------------------------------------------------
    # Control message handlers
    # -------------------------------------------------------------------------

    def _handle_register(self, request):
        """Handle a REGISTER request — map camera_id to slot."""
        existing_slot = self.camera_to_slot.get(request.camera_id)
        self.camera_to_slot[request.camera_id] = request.slot_name
        msg = f"InferenceServer-GPU{self.gpu_id}: REGISTERED {request.camera_id} -> {request.slot_name}"
        if existing_slot == request.slot_name:
            self.logger.debug(msg)
        else:
            self.logger.info(msg)
            print(msg, flush=True)

    def _handle_unregister(self, request):
        """Handle an UNREGISTER request — remove camera_id mapping."""
        self.camera_to_slot.pop(request.camera_id, None)
        msg = f"InferenceServer-GPU{self.gpu_id}: UNREGISTERED {request.camera_id}"
        self.logger.info(msg)
        print(msg, flush=True)

    # -------------------------------------------------------------------------
    # Error response helpers
    # -------------------------------------------------------------------------

    def _send_unregistered_error(self, request):
        """Log warning for requests from unregistered cameras."""
        msg = (
            f"InferenceServer-GPU{self.gpu_id}: DROPPED request from "
            f"{request.camera_id} (not registered). "
            f"Registered: {list(self.camera_to_slot.keys())}"
        )
        self.logger.warning(msg)
        print(msg, flush=True)

    def _send_error_response(self, request, slot_name, error_msg):
        """Send an error response to the camera's result queue."""
        result_queue = self.result_queues.get(slot_name)
        if result_queue:
            try:
                result_queue.put_nowait(InferenceResponse(
                    type=request.type,
                    camera_id=request.camera_id,
                    error=f"inference_server_error: {error_msg}",
                ))
            except Exception:
                pass  # Best-effort

    # -------------------------------------------------------------------------
    # Batched inference processing
    # -------------------------------------------------------------------------

    def _parse_detections(self, result):
        """Parse YOLO result object into a list of DetectionResult."""
        detections = []
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
        return detections

    def _process_detect_batch(self, requests_and_slots):
        """Process a batch of DETECT requests in one GPU call."""
        frames = [req.frame for req, _ in requests_and_slots]
        confs = [req.conf for req, _ in requests_and_slots]
        ious = [req.iou for req, _ in requests_and_slots]

        # Use minimum conf/iou across batch (conservative — no valid detections missed)
        min_conf = min(confs)
        min_iou = min(ious)

        try:
            # ONE batched GPU call for all frames
            results_list = self.model.predict(
                frames,
                conf=min_conf,
                iou=min_iou,
                classes=[0],
                half=True,
                imgsz=self.config.get('imgsz', 480),
                verbose=False
            )

            # Distribute results to per-camera result queues
            for (request, slot_name), result in zip(requests_and_slots, results_list):
                detections = self._parse_detections(result)
                response = InferenceResponse(
                    type=RequestType.DETECT,
                    camera_id=request.camera_id,
                    detections=detections,
                )
                result_queue = self.result_queues.get(slot_name)
                if result_queue:
                    try:
                        result_queue.put_nowait(response)
                    except Exception as e:
                        self.logger.error(f"Failed to deliver result for {request.camera_id}: {e}")

        except Exception as e:
            self.logger.error(f"Batch detection error: {e}")
            # On batch failure, send error to all cameras in batch
            for request, slot_name in requests_and_slots:
                self._send_error_response(request, slot_name, str(e))

    def _process_reid_batch(self, requests_and_slots):
        """Process REID requests with cross-camera batching for GPU efficiency.

        Collects all crops from all cameras in the batch, runs ONE GPU forward
        pass via extract_cross_camera_batch(), then distributes features back.
        This reduces per-camera GPU call overhead from O(N) to O(1).
        """
        if self.reid_extractor is None:
            for request, slot_name in requests_and_slots:
                response = InferenceResponse(
                    type=RequestType.REID,
                    camera_id=request.camera_id,
                    features=[None] * len(request.boxes) if request.boxes else [],
                    error="ReID extractor not available",
                )
                result_queue = self.result_queues.get(slot_name)
                if result_queue:
                    try:
                        result_queue.put_nowait(response)
                    except Exception as e:
                        self.logger.error(f"Failed to deliver REID result for {request.camera_id}: {e}")
            return

        try:
            # Collect all (frame, boxes) pairs for cross-camera batching
            frame_boxes_pairs = [
                (request.frame, request.boxes or [])
                for request, _ in requests_and_slots
            ]

            # ONE batched GPU call for all cameras' ReID crops
            all_features = self.reid_extractor.extract_cross_camera_batch(frame_boxes_pairs)

            # Distribute results to per-camera result queues
            for (request, slot_name), features in zip(requests_and_slots, all_features):
                response = InferenceResponse(
                    type=RequestType.REID,
                    camera_id=request.camera_id,
                    features=features,
                )
                result_queue = self.result_queues.get(slot_name)
                if result_queue:
                    try:
                        result_queue.put_nowait(response)
                    except Exception as e:
                        self.logger.error(f"Failed to deliver REID result for {request.camera_id}: {e}")

        except Exception as e:
            self.logger.error(f"Batch ReID error: {e}")
            for request, slot_name in requests_and_slots:
                self._send_error_response(request, slot_name, str(e))

