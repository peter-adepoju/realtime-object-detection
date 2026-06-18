"""
loader.py — Model loading utilities for YOLOv8.

This module handles:
- Downloading and caching YOLOv8 weights (auto-downloaded by Ultralytics)
- Loading models by variant name (nano, small, medium, large, xlarge)
- Providing the COCO class name list

Why a separate module?
  Model loading is repeated in multiple notebooks. Centralising it here
  ensures consistent loading logic, device selection, and error messages
  in every notebook that calls it.

Inputs:  model variant name (string), optional device string
Outputs: loaded YOLO model object, list of class name strings
"""

from pathlib import Path
from ultralytics import YOLO

# ── COCO 80-class names (ordered by class index 0–79) ────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# ── Model variant → weight filename mapping ───────────────────────────────────
MODEL_VARIANTS = {
    "nano":   "yolov8n.pt",
    "small":  "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large":  "yolov8l.pt",
    "xlarge": "yolov8x.pt",
    # Short aliases
    "n": "yolov8n.pt",
    "s": "yolov8s.pt",
    "m": "yolov8m.pt",
    "l": "yolov8l.pt",
    "x": "yolov8x.pt",
}


def load_model(variant: str = "small", models_dir: str | Path | None = None) -> YOLO:
    """
    Load a YOLOv8 model by variant name.

    Ultralytics automatically downloads the weights on first use and caches
    them in the models/ directory (or the default Ultralytics cache).

    Parameters
    ----------
    variant : str
        Model size — "nano", "small", "medium", "large", "xlarge"
        (or short aliases "n", "s", "m", "l", "x").
    models_dir : str or Path, optional
        Where to look for / save the .pt weight file.
        Defaults to <project_root>/models/.

    Returns
    -------
    YOLO
        Loaded Ultralytics YOLO model, ready for inference.

    Example
    -------
    >>> model = load_model("small")
    >>> print(model.info())
    """
    variant = variant.lower().strip()

    if variant not in MODEL_VARIANTS:
        valid = ", ".join(MODEL_VARIANTS.keys())
        raise ValueError(
            f"Unknown model variant '{variant}'. Valid options: {valid}"
        )

    weight_filename = MODEL_VARIANTS[variant]

    # Resolve models directory
    if models_dir is None:
        # Walk up to find the project root (contains requirements.txt)
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "requirements.txt").exists():
                models_dir = parent / "models"
                break
        else:
            models_dir = Path.cwd() / "models"

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    weight_path = models_dir / weight_filename

    if weight_path.exists():
        print(f"Loading cached weights from: {weight_path}")
    else:
        print(f"Downloading {weight_filename} — this happens once, then it's cached.")
        print(f"Weights will be saved to: {models_dir}")

    # YOLO() accepts both a path to existing .pt and just the filename
    # (in which case Ultralytics downloads from its release server).
    model = YOLO(str(weight_path) if weight_path.exists() else weight_filename)

    # Save to project models dir if downloaded to the default cache location
    if not weight_path.exists():
        try:
            import shutil
            # Ultralytics saves to ~/.cache/ultralytics/ by default
            import ultralytics
            cache_path = Path(ultralytics.__file__).parent / "assets" / weight_filename
            if cache_path.exists():
                shutil.copy(cache_path, weight_path)
        except Exception:
            pass  # Non-critical — model still works from cache

    print(f"Model loaded: YOLOv8 {variant.upper()} | Classes: {len(model.names)}")
    return model


def get_coco_classes() -> list[str]:
    """
    Return the ordered list of 80 COCO class names.

    Class index 0 = "person", index 1 = "bicycle", ..., index 79 = "toothbrush".

    Returns
    -------
    list of str
        80-element list of COCO category names.

    Example
    -------
    >>> classes = get_coco_classes()
    >>> print(classes[0])   # 'person'
    >>> print(classes[2])   # 'car'
    """
    return COCO_CLASSES.copy()


def get_model_info(model: YOLO) -> dict:
    """
    Extract a summary dict of model metadata useful for reporting.

    Parameters
    ----------
    model : YOLO
        Loaded Ultralytics YOLO model.

    Returns
    -------
    dict with keys:
        - n_classes (int)
        - class_names (list)
        - task (str)
        - model_name (str)
    """
    return {
        "n_classes": len(model.names),
        "class_names": list(model.names.values()),
        "task": model.task,
        "model_name": getattr(model, "model_name", "YOLOv8"),
    }
