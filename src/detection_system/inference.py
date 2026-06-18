"""
inference.py — Frame-by-frame and video-level detection inference.

This module handles:
- Single-frame detection (returns structured result dict)
- Full-video inference with optional output writing
- Confidence-threshold and class-filter support

Why a separate module?
  Inference logic (calling model.predict, parsing result objects) is
  identical across all notebooks. Centralising it avoids copy-paste bugs
  and keeps notebooks focused on analysis rather than boilerplate.

Inputs:  YOLO model, frame (numpy array) or video path
Outputs: Detection dicts, annotated frames, or saved annotated video
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


# ── Core type alias used throughout the project ───────────────────────────────
# A "detection" is a dict with these keys:
#   class_id     (int)   — COCO class index
#   class_name   (str)   — human-readable label
#   confidence   (float) — detection confidence [0, 1]
#   bbox_xyxy    (list)  — [x1, y1, x2, y2] in pixel coordinates
#   bbox_xywh    (list)  — [cx, cy, w, h] in pixel coordinates
Detection = dict


def detect_frame(
    model: YOLO,
    frame: np.ndarray,
    confidence: float = 0.25,
    classes: Optional[list[int]] = None,
    verbose: bool = False,
) -> tuple[list[Detection], float]:
    """
    Run detection on a single frame (numpy array in BGR format from cv2).

    Parameters
    ----------
    model : YOLO
        Loaded Ultralytics YOLO model.
    frame : np.ndarray
        BGR image array (H, W, 3) — the standard cv2 format.
    confidence : float
        Minimum confidence threshold. Detections below this are discarded.
    classes : list of int, optional
        If provided, only detect these COCO class IDs.
        e.g. [0, 2] detects only persons (0) and cars (2).
    verbose : bool
        If True, print Ultralytics inference summary.

    Returns
    -------
    detections : list of Detection dicts
    latency_ms : float
        Wall-clock inference time in milliseconds.

    Example
    -------
    >>> import cv2
    >>> frame = cv2.imread("frame.jpg")
    >>> detections, latency = detect_frame(model, frame, confidence=0.5)
    >>> print(f"Found {len(detections)} objects in {latency:.1f} ms")
    """
    t_start = time.perf_counter()

    results = model.predict(
        source=frame,
        conf=confidence,
        classes=classes,
        verbose=verbose,
        stream=False,
    )

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    detections = _parse_results(results)
    return detections, latency_ms


def detect_video(
    model: YOLO,
    video_path: str | Path,
    output_path: Optional[str | Path] = None,
    confidence: float = 0.25,
    classes: Optional[list[int]] = None,
    max_frames: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """
    Run detection on every frame of a video file.

    Optionally writes an annotated output video.

    Parameters
    ----------
    model : YOLO
        Loaded Ultralytics YOLO model.
    video_path : str or Path
        Path to the input video file (.mp4, .avi, etc.).
    output_path : str or Path, optional
        If provided, saves an annotated video here.
    confidence : float
        Minimum confidence threshold.
    classes : list of int, optional
        Restrict detection to specific COCO class IDs.
    max_frames : int, optional
        Stop after this many frames (useful for quick tests).
    verbose : bool
        If True, print per-frame inference logs.

    Returns
    -------
    dict with keys:
        - all_detections  : list of per-frame detection lists
        - latencies_ms    : list of per-frame latency (float)
        - frame_count     : int
        - fps_video       : float — original video FPS
        - width, height   : int — frame dimensions

    Example
    -------
    >>> results = detect_video(model, "data/raw/traffic.mp4",
    ...                        output_path="data/processed/traffic_annotated.mp4")
    >>> print(f"Processed {results['frame_count']} frames")
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video: {video_path}")

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path.name}")
    print(f"  Resolution : {width}×{height}")
    print(f"  FPS        : {fps_video:.1f}")
    print(f"  Frames     : {total_frames}")

    # Set up output writer if requested
    writer = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_video, (width, height))
        print(f"  Output     : {output_path}")

    all_detections = []
    latencies_ms = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, latency = detect_frame(
            model, frame, confidence=confidence, classes=classes, verbose=verbose
        )

        all_detections.append(detections)
        latencies_ms.append(latency)

        # Write annotated frame if output is requested
        if writer is not None:
            from .visualize import draw_detections
            annotated = draw_detections(frame.copy(), detections)
            writer.write(annotated)

        frame_idx += 1

        if verbose and frame_idx % 50 == 0:
            mean_fps = 1000.0 / (sum(latencies_ms[-50:]) / 50)
            print(f"  Frame {frame_idx}/{total_frames} | "
                  f"Mean FPS (last 50): {mean_fps:.1f}")

        if max_frames is not None and frame_idx >= max_frames:
            print(f"  Stopping after {max_frames} frames (max_frames limit).")
            break

    cap.release()
    if writer is not None:
        writer.release()

    print(f"\nFinished: {frame_idx} frames processed.")
    if latencies_ms:
        mean_lat = sum(latencies_ms) / len(latencies_ms)
        mean_fps = 1000.0 / mean_lat if mean_lat > 0 else 0
        print(f"  Mean latency : {mean_lat:.1f} ms/frame")
        print(f"  Mean FPS     : {mean_fps:.1f}")

    return {
        "all_detections": all_detections,
        "latencies_ms": latencies_ms,
        "frame_count": frame_idx,
        "fps_video": fps_video,
        "width": width,
        "height": height,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_results(results: list) -> list[Detection]:
    """
    Convert Ultralytics Results objects into our simpler list-of-dicts format.

    Each dict has:
      class_id    (int)
      class_name  (str)
      confidence  (float)
      bbox_xyxy   (list[float])   [x1, y1, x2, y2]
      bbox_xywh   (list[float])   [cx, cy, w, h]
    """
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()    # shape: (N, 4)
        boxes_xywh = result.boxes.xywh.cpu().numpy()    # shape: (N, 4)
        confs = result.boxes.conf.cpu().numpy()          # shape: (N,)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # shape: (N,)
        names = result.names                             # dict: {id: name}

        for i in range(len(class_ids)):
            detection: Detection = {
                "class_id": int(class_ids[i]),
                "class_name": names[class_ids[i]],
                "confidence": float(confs[i]),
                "bbox_xyxy": boxes_xyxy[i].tolist(),
                "bbox_xywh": boxes_xywh[i].tolist(),
            }
            detections.append(detection)

    return detections
