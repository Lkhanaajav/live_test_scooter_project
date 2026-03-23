#!/usr/bin/env python3
import os
import time
import cv2
import torch
import numpy as np
import logging
import psutil
import platform
import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List, Union
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import argparse
from stabilization import TemporalMaskSmoother

@dataclass
class Config:
    # Basic settings
    video_path: str = "test_video_june_03_3.MOV"
    model_dir: str = "models/my-segformer-road"
    output_mp4: str = "result/fast_overlay.mp4"
    road_id: int = 1
    conf_thresh: float = 0.6

    # Performance settings
    frame_step: int = 2  # Increased from 5 to 10
    use_gpu: bool = True
    enable_logging: bool = True

    # Processing settings
    enable_edge_cleaning: bool = False
    enable_simple_smoothing: bool = False  # Simpler temporal smoothing
    smoothing_weight: float = 0.2  # Weight for previous frame (0.2 = 20% previous, 80% current)
    inference_resize: Optional[Tuple[int, int]] = None  # (width, height) override for SegFormer input

    # ONNX acceleration
    onnx_model_path: Optional[str] = None  # Path to .onnx model (auto-detected if None)

@dataclass
class SystemInfo:
    cpu: str = ""
    gpu: str = ""
    ram: float = 0.0  # GB
    os: str = ""
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""

@dataclass
class PerformanceMetrics:
    fps: float = 0.0
    inference_time: float = 0.0  # ms per frame
    memory_usage: Dict[str, float] = None  # MB
    frame_count: int = 0
    processed_count: int = 0
    total_time: float = 0.0  # seconds
    processing_start_time: float = 0.0  # When actual processing begins
    processing_end_time: float = 0.0    # When processing ends
    pure_processing_time: float = 0.0   # Only processing time (excluding setup)

def get_gpu_memory():
    """Get GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2, torch.cuda.memory_reserved() / 1024**2
    return 0, 0

def get_cpu_memory():
    """Get CPU memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

@dataclass
class MemoryStats:
    gpu_allocated: float = 0  # MB
    gpu_reserved: float = 0   # MB
    cpu_used: float = 0       # MB
    timestamp: float = 0      # seconds

def get_system_info() -> SystemInfo:
    """Get detailed system information."""
    info = SystemInfo()
    
    # CPU Info
    info.cpu = platform.processor()
    
    # GPU Info
    if torch.cuda.is_available():
        info.gpu = torch.cuda.get_device_name(0)
        info.cuda_version = torch.version.cuda
    
    # RAM Info
    info.ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
    
    # OS Info
    info.os = platform.platform()
    
    # Python and PyTorch versions
    info.python_version = platform.python_version()
    info.torch_version = torch.__version__
    
    return info

class FastRoadDetector:
    def __init__(self, config: Config):
        self.config = config
        self.memory_stats = []
        self.performance_metrics = PerformanceMetrics()
        self.system_info = get_system_info()
        self._setup_logging()
        self._setup_device()
        self._ort_session = None  # ONNX Runtime session (None = use PyTorch)
        self._load_model()
        self.previous_mask = None
        
    def _setup_logging(self):
        if self.config.enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())

    def _setup_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            # Set CUDA device properties for better performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Clear CUDA cache
            torch.cuda.empty_cache()
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            self.logger.warning("GPU not available, using CPU instead")

    def _log_memory_usage(self, stage: str):
        """Log current memory usage."""
        gpu_alloc, gpu_reserved = get_gpu_memory()
        cpu_used = get_cpu_memory()
        
        stats = MemoryStats(
            gpu_allocated=gpu_alloc,
            gpu_reserved=gpu_reserved,
            cpu_used=cpu_used,
            timestamp=time.time()
        )
        self.memory_stats.append(stats)
        
        self.logger.info(f"Memory Usage ({stage}):")
        if torch.cuda.is_available():
            self.logger.info(f"  GPU Allocated: {gpu_alloc:.2f} MB")
            self.logger.info(f"  GPU Reserved:  {gpu_reserved:.2f} MB")
        self.logger.info(f"  CPU Used:      {cpu_used:.2f} MB")

    @staticmethod
    def _resolve_model_dir(model_dir: str) -> str:
        """Resolve relative model paths from this file's directory."""
        if not os.path.isabs(model_dir):
            local_candidate = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                model_dir
            )
            if os.path.isdir(local_candidate):
                return local_candidate
        return model_dir

    def _find_onnx_model(self, model_dir: str) -> Optional[str]:
        """Auto-detect ONNX model file next to the PyTorch model."""
        if self.config.onnx_model_path:
            path = self.config.onnx_model_path
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
            return path if os.path.isfile(path) else None

        # Check for ONNX export directory (model_dir + "-onnx")
        # Prefer FP32 — INT8 dynamic quantization can be slower on some CPUs
        onnx_dir = model_dir.rstrip("/\\") + "-onnx"
        for name in ("model.onnx", "model_int8.onnx"):
            candidate = os.path.join(onnx_dir, name)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _load_model(self):
        try:
            self._log_memory_usage("Before model load")
            model_dir = self._resolve_model_dir(self.config.model_dir)

            # Try ONNX Runtime first (much faster on CPU)
            onnx_path = self._find_onnx_model(model_dir)
            if onnx_path is not None:
                self._load_onnx_model(onnx_path, model_dir)
            else:
                self._load_pytorch_model(model_dir)

            self._log_memory_usage("After model load")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def _load_onnx_model(self, onnx_path: str, model_dir: str):
        """Load ONNX model via ONNX Runtime for fast CPU inference."""
        import onnxruntime as ort

        self.logger.info(f"Loading ONNX model: {onnx_path}")
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.inter_op_num_threads = 2
        sess_opts.intra_op_num_threads = 4
        self._ort_session = ort.InferenceSession(
            onnx_path, sess_opts, providers=["CPUExecutionProvider"]
        )
        # We still need the image processor for normalization
        if os.path.isdir(model_dir):
            self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        else:
            self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = None  # No PyTorch model needed
        self.logger.info(f"ONNX Runtime ready ({os.path.basename(onnx_path)})")

    def _load_pytorch_model(self, model_dir: str):
        """Load PyTorch model (fallback when no ONNX available)."""
        if os.path.isdir(model_dir):
            self.logger.info(f"Loading model from local directory: {model_dir}")
            self.processor = AutoImageProcessor.from_pretrained(
                model_dir,
                local_files_only=True
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_dir,
                local_files_only=True
            ).to(self.device)
        else:
            self.logger.warning(
                f"Model path not found locally: {model_dir}. "
                "Trying HuggingFace identifier resolution."
            )
            self.processor = AutoImageProcessor.from_pretrained(model_dir)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_dir
            ).to(self.device)
        self.model.eval()
        self.logger.info("PyTorch model loaded successfully")

    def _apply_simple_smoothing(self, current_mask: np.ndarray) -> np.ndarray:
        """Apply simple temporal smoothing without optical flow."""
        if not self.config.enable_simple_smoothing or self.previous_mask is None:
            return current_mask
        
        # Simple weighted average with previous mask
        return cv2.addWeighted(
            current_mask, 
            1 - self.config.smoothing_weight,
            self.previous_mask,
            self.config.smoothing_weight,
            0
        )

    def _clean_mask(self, mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Optimized mask cleaning with improved smoothness and accuracy."""
        if not self.config.enable_edge_cleaning:
            return mask
            
        # Edge detection with smaller kernel for speed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)  # Reduced kernel size
        edges = cv2.Canny(blur, 50, 150)
        barrier = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))  # Reduced kernel size
        
        # Remove edge crossings
        mask[barrier > 0] = 0
        
        # Remove small islands (keep largest component)
        num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_lbl > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = 1 + np.argmax(areas)
            mask = np.where(labels == largest, 255, 0).astype(np.uint8)
        
        # Smooth the mask while preserving important features
        # 1. Initial smoothing to reduce noise
        mask = cv2.GaussianBlur(mask, (3,3), 0.5)
        
        # 2. Remove small artifacts while preserving shape
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # 3. Fill small holes while preserving important gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 4. Bilateral filter for edge-preserving smoothing
        mask = cv2.bilateralFilter(mask.astype(np.float32), 5, 75, 75)
        
        # 5. Final threshold to get binary mask
        mask = (mask > 127).astype(np.uint8) * 255
        
        # 6. Final light smoothing of boundaries
        mask = cv2.GaussianBlur(mask, (3,3), 0.3)  # Very light smoothing
        mask = (mask > 127).astype(np.uint8) * 255
            
        return mask

    def _log_system_info(self):
        """Log system information."""
        self.logger.info("\nSystem Information:")
        self.logger.info(f"CPU: {self.system_info.cpu}")
        self.logger.info(f"GPU: {self.system_info.gpu}")
        self.logger.info(f"RAM: {self.system_info.ram:.2f} GB")
        self.logger.info(f"OS: {self.system_info.os}")
        self.logger.info(f"Python Version: {self.system_info.python_version}")
        self.logger.info(f"PyTorch Version: {self.system_info.torch_version}")
        self.logger.info(f"CUDA Version: {self.system_info.cuda_version}")

    @staticmethod
    def _normalize_processor_size(size: Union[int, Tuple[int, int], List[int], Dict[str, int]]) -> Union[int, Dict[str, int]]:
        if isinstance(size, int):
            return int(size)
        if isinstance(size, dict):
            out: Dict[str, int] = {}
            if "height" in size:
                out["height"] = int(size["height"])
            if "width" in size:
                out["width"] = int(size["width"])
            if not out:
                raise ValueError("Processor size dict must contain 'height' and/or 'width'.")
            return out
        if isinstance(size, (tuple, list)) and len(size) == 2:
            width, height = size
            return {"height": int(height), "width": int(width)}
        raise ValueError(f"Unsupported processor size specification: {size}")

    def process_frame(
        self,
        frame: np.ndarray,
        processor_size: Optional[Union[int, Tuple[int, int], List[int], Dict[str, int]]] = None,
        return_overlay: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Process a single frame and return the segmentation mask and optional overlay."""
        start_time = time.time()

        H, W = frame.shape[:2]
        size_override = processor_size if processor_size is not None else self.config.inference_resize

        if self._ort_session is not None:
            # --- ONNX Runtime inference (fast CPU path) ---
            # Direct OpenCV preprocessing — no HuggingFace processor overhead
            if size_override is not None:
                sz = self._normalize_processor_size(size_override)
                if isinstance(sz, dict):
                    target_h, target_w = sz.get("height", 256), sz.get("width", 256)
                else:
                    target_h = target_w = sz
            else:
                target_h = target_w = 512
            # BGR->RGB + resize in one step
            rgb_resized = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )
            # ImageNet normalization
            pixel_values = rgb_resized.astype(np.float32) * np.float32(1.0 / 255.0)
            pixel_values -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
            pixel_values /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
            pixel_values = np.ascontiguousarray(pixel_values.transpose(2, 0, 1)[np.newaxis])

            logits = self._ort_session.run(None, {"pixel_values": pixel_values})[0]
            # Softmax at low-res (much faster than full frame)
            logits_2d = logits[0]  # (C, h, w)
            exp = np.exp(logits_2d - logits_2d.max(axis=0, keepdims=True))
            probs_lowres = (exp / exp.sum(axis=0, keepdims=True))[self.config.road_id]
            mask_lowres = (probs_lowres > self.config.conf_thresh).astype(np.uint8) * 255
            mask = cv2.resize(mask_lowres, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            # --- PyTorch inference (original path) ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processor_kwargs = {"return_tensors": "pt"}
            if size_override is not None:
                processor_kwargs["size"] = self._normalize_processor_size(size_override)
            inputs = self.processor(rgb, **processor_kwargs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
            up = torch.nn.functional.interpolate(
                logits, size=(H, W), mode="bilinear", align_corners=False
            )[0]
            probs = up.softmax(0)[self.config.road_id].cpu().numpy()
            mask = (probs > self.config.conf_thresh).astype(np.uint8) * 255

        # Clean mask
        mask = self._clean_mask(mask, frame)

        # Apply temporal smoothing
        mask = self._apply_simple_smoothing(mask)
        self.previous_mask = mask.copy()

        overlay = None
        if return_overlay:
            overlay = frame.copy()
            overlay[mask == 255] = (0, 255, 0)

        # Update metrics
        inference_time = (time.time() - start_time) * 1000
        self.performance_metrics.inference_time = (
            self.performance_metrics.inference_time * self.performance_metrics.processed_count + inference_time
        ) / (self.performance_metrics.processed_count + 1)
        self.performance_metrics.processed_count += 1

        return mask, overlay

    def process_video(self):
        """Process the entire video with the configured settings."""
        # Setup video capture
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.config.video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        os.makedirs(os.path.dirname(self.config.output_mp4), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vis = cv2.VideoWriter(self.config.output_mp4, fourcc, fps, (W, H), isColor=True)

        # Processing loop
        frame_idx = 0
        processed_idx = 0
        inference_times = []
        _smoother = TemporalMaskSmoother()
        _last_smoothed_mask = None
        _prev_gray = None
        _flow_feat = dict(maxCorners=100, qualityLevel=0.01, minDistance=30, blockSize=3)
        _flow_lk = dict(winSize=(15, 15), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Start timing when we begin processing
        self.performance_metrics.processing_start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.config.frame_step == 0:
                    # Start timing for this frame
                    frame_start = time.time()

                    # Minimal optical flow between processed frames only
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    flow_dy = 0.0
                    if _prev_gray is not None:
                        pts = cv2.goodFeaturesToTrack(_prev_gray, **_flow_feat)
                        if pts is not None and len(pts) >= 6:
                            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                                _prev_gray, gray, pts, None, **_flow_lk)
                            good = (status.ravel() == 1)
                            if good.sum() >= 6:
                                m, _ = cv2.estimateAffinePartial2D(
                                    pts[good], curr_pts[good])
                                if m is not None:
                                    flow_dy = float(m[1, 2])
                    _prev_gray = gray

                    mask, _ = self.process_frame(frame)
                    smoothed_mask = _smoother.smooth(mask, flow_dy=flow_dy)
                    _last_smoothed_mask = smoothed_mask
                    self.previous_mask = smoothed_mask
                    processed_idx += 1

                    # Rebuild overlay from smoothed mask
                    overlay = frame.copy()
                    overlay[smoothed_mask == 255] = (0, 255, 0)

                    # Record inference time
                    inference_time = (time.time() - frame_start) * 1000  # Convert to ms
                    inference_times.append(inference_time)
                else:
                    overlay = frame.copy()
                    if _last_smoothed_mask is not None:
                        overlay[_last_smoothed_mask == 255] = (0, 255, 0)

                # Preview
                preview = cv2.resize(overlay, (W//2, H//2))
                cv2.imshow("Fast Road Detection", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                out_vis.write(overlay)
                frame_idx += 1

        finally:
            # End timing when processing is complete
            self.performance_metrics.processing_end_time = time.time()
            self.performance_metrics.pure_processing_time = (
                self.performance_metrics.processing_end_time - 
                self.performance_metrics.processing_start_time
            )
            
            # Cleanup
            cap.release()
            out_vis.release()
            cv2.destroyAllWindows()

        # Update performance metrics
        self.performance_metrics.frame_count = frame_idx
        self.performance_metrics.processed_count = processed_idx
        self.performance_metrics.inference_time = np.mean(inference_times) if inference_times else 0
        self.performance_metrics.fps = processed_idx / self.performance_metrics.pure_processing_time if self.performance_metrics.pure_processing_time > 0 else 0
        
        # Log final memory usage
        self._log_memory_usage("After processing")
        
        # Log performance summary
        self._log_performance_summary()
        
        # Save metrics
        self._save_metrics()

    def _log_performance_summary(self):
        """Log detailed performance summary."""
        self.logger.info("\nPerformance Summary:")
        self.logger.info(f"Total Frames: {self.performance_metrics.frame_count}")
        self.logger.info(f"Processed Frames: {self.performance_metrics.processed_count}")
        self.logger.info(f"Pure Processing Time: {self.performance_metrics.pure_processing_time:.2f} seconds")
        self.logger.info(f"Average Inference Time: {self.performance_metrics.inference_time:.2f} ms")
        self.logger.info(f"Processing FPS: {self.performance_metrics.fps:.2f}")
        
        if torch.cuda.is_available():
            gpu_alloc, gpu_reserved = get_gpu_memory()
            self.logger.info(f"Final GPU Memory - Allocated: {gpu_alloc:.2f} MB, Reserved: {gpu_reserved:.2f} MB")
        
        cpu_used = get_cpu_memory()
        self.logger.info(f"Final CPU Memory Usage: {cpu_used:.2f} MB")

    def _save_metrics(self):
        """Save performance metrics to a JSON file."""
        metrics = {
            "system_info": asdict(self.system_info),
            "performance_metrics": asdict(self.performance_metrics),
            "config": asdict(self.config)
        }
        
        # Create metrics directory if it doesn't exist
        os.makedirs("metrics", exist_ok=True)
        
        # Save to file with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"metrics/performance_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        self.logger.info(f"\nMetrics saved to {filename}")

def toggle_device(detector: FastRoadDetector, use_gpu: bool) -> FastRoadDetector:
    """Quickly toggle between CPU and GPU usage."""
    config = detector.config
    config.use_gpu = use_gpu
    return FastRoadDetector(config)

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Road Detection with GPU/CPU toggle')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for processing')
    parser.add_argument('--use-cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--video', type=str, default="test_video_june_03_1.MOV", help='Input video path')
    parser.add_argument('--output', type=str, default="result/fast_overlay.mp4", help='Output video path')
    parser.add_argument('--save-metrics', action='store_true', help='Save detailed performance metrics')
    
    args = parser.parse_args()
    
    # Determine device based on arguments
    use_gpu = args.use_gpu and not args.use_cpu
    
    # Create configuration
    config = Config(
        video_path=args.video,
        output_mp4=args.output,
        use_gpu=use_gpu
    )
    
    # Create and run detector
    detector = FastRoadDetector(config)
    detector.process_video()

if __name__ == "__main__":
    main() 