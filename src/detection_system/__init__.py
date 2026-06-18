"""
detection_system — core library for the Real-Time Object Detection project.

Modules
-------
loader      : Model loading (YOLOv8 weights, COCO class names)
inference   : Frame-by-frame detection inference
counting    : Per-class object counting logic
benchmark   : Latency, FPS, and memory profiling
visualize   : Bounding-box drawing and annotation utilities
utils       : Shared helpers — figure saving, path management, config loading
"""

# Core modules that do NOT require ultralytics — always importable
from .counting import count_objects, count_objects_over_time
from .visualize import draw_detections, annotate_frame
from .utils import save_figure, save_table, get_project_root, load_config

# Modules that require ultralytics — imported lazily so tests
# can run without the full deep-learning stack installed.
def __getattr__(name):
    if name in ("load_model", "get_coco_classes"):
        from .loader import load_model, get_coco_classes  # noqa: F401
        return locals()[name]
    if name in ("detect_frame", "detect_video"):
        from .inference import detect_frame, detect_video  # noqa: F401
        return locals()[name]
    if name in ("BenchmarkResult", "benchmark_model"):
        from .benchmark import BenchmarkResult, benchmark_model  # noqa: F401
        return locals()[name]
    raise AttributeError(f"module 'detection_system' has no attribute {name!r}")

__version__ = "1.0.0"
__author__ = "P. O. Adepoju"
