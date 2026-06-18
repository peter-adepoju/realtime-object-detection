"""
visualize.py — Bounding-box drawing and frame annotation utilities.

This module handles:
- Drawing bounding boxes and class labels on frames
- Overlaying per-class count summaries on frames
- Preparing frames for display in notebooks (BGR → RGB conversion)
- Colorblind-safe class color palette

Why a separate module?
  Drawing logic uses cv2 functions that are verbose and repeated constantly.
  Centralising ensures consistent visual style (font, thickness, colors)
  across all notebooks and the Streamlit app.

Inputs:  BGR numpy frames + detection dicts
Outputs: Annotated BGR numpy frames (can be saved or displayed)
"""

from typing import Optional

import cv2
import numpy as np

# Detection is a plain dict type alias — no ultralytics dependency needed here
Detection = dict


# ── Class color palette (BGR format for cv2) ──────────────────────────────────
# Using a colorblind-safe 10-color palette derived from Tableau 10
# Defined as BGR tuples (cv2 uses BGR, not RGB)
_PALETTE_BGR = [
    (86, 180, 233),    # sky blue
    (230, 159, 0),     # orange
    (0, 158, 115),     # bluish green
    (204, 121, 167),   # reddish purple
    (0, 114, 178),     # blue
    (213, 94, 0),      # vermilion
    (240, 228, 66),    # yellow
    (0, 0, 0),         # black (fallback)
    (220, 220, 220),   # light grey
    (145, 30, 180),    # purple
]


def _get_color(class_id: int) -> tuple[int, int, int]:
    """Return a consistent BGR color for a given class ID."""
    return _PALETTE_BGR[class_id % len(_PALETTE_BGR)]


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    show_confidence: bool = True,
    box_thickness: int = 2,
    font_scale: float = 0.55,
    label_padding: int = 4,
) -> np.ndarray:
    """
    Draw bounding boxes and class labels on a BGR frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image array (H, W, 3). This is MODIFIED IN PLACE — pass a copy
        if you need to preserve the original.
    detections : list of Detection dicts
        Output of detect_frame().
    show_confidence : bool
        If True, append confidence score to the label text.
    box_thickness : int
        Bounding box line thickness in pixels.
    font_scale : float
        cv2 font scale for label text.
    label_padding : int
        Pixel padding around label text background rectangle.

    Returns
    -------
    np.ndarray
        Annotated BGR frame (same array, modified in place and returned).

    Example
    -------
    >>> annotated = draw_detections(frame.copy(), detections)
    >>> # Display in notebook:
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    >>> plt.axis("off"); plt.show()
    """
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        cls_id = det["class_id"]
        cls_name = det["class_name"]
        conf = det["confidence"]

        color = _get_color(cls_id)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

        # Build label text
        label = cls_name if not show_confidence else f"{cls_name} {conf:.2f}"

        # Label background rectangle
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        label_bg_y1 = max(y1 - text_h - 2 * label_padding, 0)
        label_bg_y2 = y1
        cv2.rectangle(
            frame,
            (x1, label_bg_y1),
            (x1 + text_w + 2 * label_padding, label_bg_y2),
            color,
            -1,  # filled
        )

        # Label text (white for visibility)
        cv2.putText(
            frame,
            label,
            (x1 + label_padding, y1 - label_padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def annotate_frame(
    frame: np.ndarray,
    detections: list[Detection],
    frame_idx: Optional[int] = None,
    show_count_overlay: bool = True,
) -> np.ndarray:
    """
    Full annotation pipeline: draw boxes + count overlay on a frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (will be modified in place — pass a copy if needed).
    detections : list of Detection dicts
    frame_idx : int, optional
        Frame number to display in overlay.
    show_count_overlay : bool
        If True, adds a count summary panel in the top-right corner.

    Returns
    -------
    np.ndarray
        Fully annotated BGR frame.

    Example
    -------
    >>> annotated = annotate_frame(frame.copy(), detections, frame_idx=42)
    """
    annotated = draw_detections(frame, detections)

    if show_count_overlay:
        annotated = _draw_count_overlay(annotated, detections)

    if frame_idx is not None:
        cv2.putText(
            annotated,
            f"Frame: {frame_idx}",
            (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    return annotated


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert a BGR frame (cv2 default) to RGB (matplotlib default).

    This is called before plt.imshow() in every notebook that displays
    frames. cv2 reads images as BGR; matplotlib shows them as RGB.
    Forgetting this conversion causes images to appear with wrong colors.

    Parameters
    ----------
    frame : np.ndarray
        BGR image array.

    Returns
    -------
    np.ndarray
        RGB image array.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_frame(frame: np.ndarray, max_width: int = 960) -> np.ndarray:
    """
    Resize a frame to at most max_width pixels wide, preserving aspect ratio.

    Useful for notebooks where full-resolution frames are too large.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR frame.
    max_width : int
        Maximum allowed width.

    Returns
    -------
    np.ndarray
        Resized frame (or original if already within limit).
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _draw_count_overlay(
    frame: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """Draw a per-class count summary panel in the top-right corner."""
    from collections import Counter

    counts = Counter(d["class_name"] for d in detections)
    if not counts:
        return frame

    h, w = frame.shape[:2]
    line_height = 22
    panel_h = len(counts) * line_height + 16
    panel_w = 200
    x_start = w - panel_w - 10
    y_start = 10

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x_start, y_start),
        (x_start + panel_w, y_start + panel_h),
        (30, 30, 30),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Class counts text
    for i, (cls_name, cnt) in enumerate(counts.most_common()):
        text = f"{cls_name}: {cnt}"
        y_pos = y_start + 16 + i * line_height
        cv2.putText(
            frame,
            text,
            (x_start + 8, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame
