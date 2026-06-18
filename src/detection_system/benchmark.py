"""
benchmark.py — Engineering performance profiling for YOLOv8 inference.

This module measures:
- Per-frame inference latency (ms)
- Throughput (FPS) — sustained over many frames
- Peak memory usage (MB) — RAM during inference
- Model file size (MB) — on-disk weight size

Why a separate module?
  Benchmarking requires careful timing (perf_counter, not time.time),
  memory snapshotting (psutil), and warm-up passes. Centralising this
  prevents subtle measurement errors in the notebooks.

Inputs:  YOLO model, a set of test frames
Outputs: BenchmarkResult dataclass, summary DataFrame
"""

import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psutil

from ultralytics import YOLO


@dataclass
class BenchmarkResult:
    """
    Container for all benchmark metrics collected for one model variant.

    Attributes
    ----------
    model_name : str
        Human-readable model name, e.g. "YOLOv8-small".
    weight_file : str
        Filename of the .pt weight file.
    model_size_mb : float
        Size of the weight file on disk in megabytes.
    n_frames : int
        Number of frames used in the benchmark run.
    latencies_ms : list of float
        Per-frame inference latency in milliseconds.
    peak_memory_mb : float
        Peak RAM usage (MB) during the benchmark.
    frame_shape : tuple
        (H, W, C) of the benchmark frames.
    """
    model_name: str
    weight_file: str
    model_size_mb: float
    n_frames: int
    latencies_ms: list[float]
    peak_memory_mb: float
    frame_shape: tuple
    extra: dict = field(default_factory=dict)

    # ── Derived statistics (computed from latencies_ms) ───────────────────────

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms))

    @property
    def median_latency_ms(self) -> float:
        return float(np.median(self.latencies_ms))

    @property
    def p95_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95))

    @property
    def p99_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99))

    @property
    def std_latency_ms(self) -> float:
        return float(np.std(self.latencies_ms))

    @property
    def mean_fps(self) -> float:
        return 1000.0 / self.mean_latency_ms if self.mean_latency_ms > 0 else 0.0

    @property
    def min_latency_ms(self) -> float:
        return float(np.min(self.latencies_ms))

    @property
    def max_latency_ms(self) -> float:
        return float(np.max(self.latencies_ms))

    def to_dict(self) -> dict:
        """Flat dict for DataFrame construction or CSV export."""
        return {
            "model_name": self.model_name,
            "weight_file": self.weight_file,
            "model_size_mb": round(self.model_size_mb, 2),
            "n_frames": self.n_frames,
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "median_latency_ms": round(self.median_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "std_latency_ms": round(self.std_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "mean_fps": round(self.mean_fps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "frame_height": self.frame_shape[0],
            "frame_width": self.frame_shape[1],
        }


def get_model_size_mb(model_path: str | Path) -> float:
    """Return the on-disk size of a .pt file in megabytes."""
    path = Path(model_path)
    if path.exists():
        return path.stat().st_size / (1024 ** 2)
    return float("nan")


def benchmark_model(
    model: YOLO,
    frames: list[np.ndarray],
    model_name: str = "YOLOv8",
    weight_path: Optional[str | Path] = None,
    confidence: float = 0.25,
    n_warmup: int = 5,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Measure inference latency, FPS, and memory for a YOLO model.

    The first n_warmup frames are discarded — they include model JIT
    compilation overhead that would inflate latency measurements.

    Parameters
    ----------
    model : YOLO
        Loaded Ultralytics YOLO model.
    frames : list of np.ndarray
        BGR frames to benchmark on (from cv2.VideoCapture or mock data).
        At least n_warmup + 10 frames recommended.
    model_name : str
        Label for this model (used in reports).
    weight_path : str or Path, optional
        Path to the .pt file — used to measure model size on disk.
    confidence : float
        Detection confidence threshold (same as used in real inference).
    n_warmup : int
        Number of warm-up frames to discard before measuring.
    verbose : bool
        If True, print progress and final summary.

    Returns
    -------
    BenchmarkResult
        Dataclass containing all latency statistics and metadata.

    Example
    -------
    >>> result = benchmark_model(model, frames, model_name="YOLOv8-small")
    >>> print(f"Mean FPS: {result.mean_fps:.1f}")
    >>> df = pd.DataFrame([result.to_dict()])
    """
    if len(frames) == 0:
        raise ValueError("No frames provided for benchmarking.")

    if len(frames) <= n_warmup:
        raise ValueError(
            f"Need more than n_warmup ({n_warmup}) frames. "
            f"Got {len(frames)} frames."
        )

    frame_shape = frames[0].shape

    if verbose:
        print(f"Benchmarking: {model_name}")
        print(f"  Frames     : {len(frames)} (first {n_warmup} are warm-up)")
        print(f"  Frame size : {frame_shape[1]}×{frame_shape[0]} px")
        print(f"  Confidence : {confidence}")

    # Warm-up — discard these results (they include JIT overhead)
    if verbose:
        print(f"  Running {n_warmup} warm-up frames...")
    for frame in frames[:n_warmup]:
        _ = model.predict(source=frame, conf=confidence, verbose=False)

    # GC before measuring memory
    gc.collect()
    process = psutil.Process(os.getpid())
    mem_before_mb = process.memory_info().rss / (1024 ** 2)

    # Benchmark frames
    latencies_ms = []
    peak_mem_mb = mem_before_mb

    for i, frame in enumerate(frames[n_warmup:]):
        t_start = time.perf_counter()
        _ = model.predict(source=frame, conf=confidence, verbose=False)
        t_end = time.perf_counter()

        latencies_ms.append((t_end - t_start) * 1000.0)

        current_mem = process.memory_info().rss / (1024 ** 2)
        if current_mem > peak_mem_mb:
            peak_mem_mb = current_mem

        if verbose and (i + 1) % 20 == 0:
            mean_so_far = sum(latencies_ms) / len(latencies_ms)
            print(f"  Frame {i + 1}/{len(frames) - n_warmup} | "
                  f"Running mean: {mean_so_far:.1f} ms ({1000/mean_so_far:.1f} FPS)")

    # Model size
    model_size_mb = float("nan")
    if weight_path is not None:
        model_size_mb = get_model_size_mb(weight_path)

    result = BenchmarkResult(
        model_name=model_name,
        weight_file=str(weight_path) if weight_path else "unknown",
        model_size_mb=model_size_mb,
        n_frames=len(latencies_ms),
        latencies_ms=latencies_ms,
        peak_memory_mb=peak_mem_mb - mem_before_mb,
        frame_shape=frame_shape,
    )

    if verbose:
        print(f"\n  ── Results for {model_name} ──")
        print(f"  Mean latency   : {result.mean_latency_ms:.1f} ms")
        print(f"  Mean FPS       : {result.mean_fps:.1f}")
        print(f"  p95 latency    : {result.p95_latency_ms:.1f} ms")
        print(f"  p99 latency    : {result.p99_latency_ms:.1f} ms")
        print(f"  Peak mem (Δ)   : {result.peak_memory_mb:.1f} MB")
        if not np.isnan(model_size_mb):
            print(f"  Model size     : {model_size_mb:.1f} MB")

    return result


def compare_benchmarks(results: list[BenchmarkResult]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a list of BenchmarkResult objects.

    Parameters
    ----------
    results : list of BenchmarkResult
        One result per model variant (e.g. nano, small, medium).

    Returns
    -------
    pd.DataFrame
        One row per model, columns = all benchmark metrics.
        Ready for CSV export and matplotlib plotting.

    Example
    -------
    >>> df = compare_benchmarks([nano_result, small_result, medium_result])
    >>> df.to_csv("reports/tables/benchmark_comparison.csv", index=False)
    >>> df[["model_name", "mean_fps", "model_size_mb"]].plot(x="model_name")
    """
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)
    df = df.sort_values("mean_fps", ascending=False).reset_index(drop=True)
    return df
