"""
counting.py — Object counting logic for detection results.

This module handles:
- Per-frame object counting (how many of each class in one frame)
- Aggregation over a full video (total appearances, mean counts, etc.)
- Building a time-series DataFrame of counts across frames

Why a separate module?
  Counting is conceptually distinct from detection. Keeping it separate
  makes it easy to test independently and swap in different counting
  strategies (e.g., centroid-tracking-based counting) without touching
  the inference code.

Inputs:  Detection dicts (from inference.py) or lists of them
Outputs: Count dicts, pandas DataFrames of time-series counts
"""

from collections import Counter
from typing import Optional

import pandas as pd

# Detection is just a plain dict — we use it only as a type alias for documentation.
# We do NOT import from inference at module level to keep counting independent
# of the ultralytics stack (so tests can run without deep-learning dependencies).
Detection = dict  # class_id, class_name, confidence, bbox_xyxy, bbox_xywh


def count_objects(detections: list) -> dict[str, int]:
    """
    Count the number of detections per class in a single frame.

    Parameters
    ----------
    detections : list of Detection dicts
        Output of detect_frame() for one frame.

    Returns
    -------
    dict mapping class_name → count (int)
        Classes with zero detections are not included.

    Example
    -------
    >>> counts = count_objects(detections)
    >>> print(counts)  # {'person': 3, 'car': 5}
    """
    counter = Counter(d["class_name"] for d in detections)
    return dict(counter)


def count_objects_over_time(
    all_detections: list,
    all_class_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Build a per-frame count DataFrame from a list of per-frame detection lists.

    One row per frame, one column per class. Missing classes = 0.

    Parameters
    ----------
    all_detections : list of list of Detection
        Outer list: one entry per frame.
        Inner list: detections for that frame (from detect_frame()).
    all_class_names : list of str, optional
        Full set of class names to include as columns. If None, uses the
        union of all class names that appear in the detections.

    Returns
    -------
    pd.DataFrame
        Shape: (n_frames, n_classes)
        Index: frame number (0-based)
        Columns: class names (sorted alphabetically)
        Values: integer counts per class per frame

    Example
    -------
    >>> df = count_objects_over_time(results["all_detections"])
    >>> df.head()
    >>> df["person"].plot(title="Person count over time")
    """
    per_frame_counts = [count_objects(frame_dets) for frame_dets in all_detections]

    # Build the full class set
    if all_class_names is None:
        all_class_names = sorted(
            {cls for frame in per_frame_counts for cls in frame.keys()}
        )

    # Fill a DataFrame — missing classes become 0
    df = pd.DataFrame(per_frame_counts, columns=all_class_names).fillna(0).astype(int)
    df.index.name = "frame"

    return df


def summarise_counts(count_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics across all frames for each detected class.

    Parameters
    ----------
    count_df : pd.DataFrame
        Output of count_objects_over_time().

    Returns
    -------
    pd.DataFrame
        One row per class, columns:
          - total_appearances   : sum of count across all frames
          - mean_per_frame      : mean count per frame (incl. frames with 0)
          - max_in_single_frame : peak count in any single frame
          - frames_present      : number of frames where count > 0
          - pct_frames_present  : frames_present / total_frames * 100

    Example
    -------
    >>> summary = summarise_counts(count_df)
    >>> print(summary.sort_values("total_appearances", ascending=False))
    """
    n_frames = len(count_df)

    summary = pd.DataFrame({
        "total_appearances": count_df.sum(),
        "mean_per_frame": count_df.mean().round(2),
        "max_in_single_frame": count_df.max(),
        "frames_present": (count_df > 0).sum(),
        "pct_frames_present": ((count_df > 0).sum() / n_frames * 100).round(1),
    })

    # Remove classes that never appeared
    summary = summary[summary["total_appearances"] > 0]
    summary = summary.sort_values("total_appearances", ascending=False)
    summary.index.name = "class_name"

    return summary


def get_dominant_classes(count_df: pd.DataFrame, top_n: int = 5) -> list[str]:
    """
    Return the top_n most frequently detected class names.

    Useful for plotting — avoids cluttering figures with rare classes.

    Parameters
    ----------
    count_df : pd.DataFrame
        Output of count_objects_over_time().
    top_n : int
        How many classes to return.

    Returns
    -------
    list of str
        Class names sorted by total_appearances (descending), up to top_n.

    Example
    -------
    >>> top5 = get_dominant_classes(count_df, top_n=5)
    >>> count_df[top5].plot()
    """
    totals = count_df.sum().sort_values(ascending=False)
    return list(totals.head(top_n).index)
